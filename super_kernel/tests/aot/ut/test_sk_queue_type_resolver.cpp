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

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_queue_type_resolver.h"
#include "sk_task_ir.h"

class SkQueueTypeResolverTest : public testing::Test {
 protected:
  SuperKernelBaseNode *AddKernel(uint64_t id, uint32_t stream, uint64_t previous, uint64_t next, SkKernelType type) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, stream, INVALID_STREAM_ID,
                                                        previous);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetNodeId(id);
    node->SetPreNodeId(previous);
    node->SetNextNodeId(next);
    node->nodeInfos.kernelInfos.kernelType = type;
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
    auto *result = node.get();
    graph.graphMap[id] = std::move(node);
    return result;
  }

  SuperKernelGraph graph;
};

TEST_F(SkQueueTypeResolverTest, ReturnsQueuesInTaskOrder) {
  auto *aic = AddKernel(1, 0, INVALID_TASK_ID, 2, SkKernelType::AIC_ONLY);
  auto *notify = AddEvent(2, 0, 1, INVALID_TASK_ID, SkNodeType::NODE_NOTIFY);
  auto *wait = AddEvent(3, 1, INVALID_TASK_ID, 4, SkNodeType::NODE_WAIT);
  auto *aiv = AddKernel(4, 1, 3, INVALID_TASK_ID, SkKernelType::AIV_ONLY);

  std::vector<SkQueueType> queues;
  ASSERT_TRUE(SkQueueTypeResolver(graph).Resolve({aic, notify, wait, aiv}, queues));
  EXPECT_EQ(queues, (std::vector<SkQueueType>{SkQueueType::AIC, SkQueueType::AIC, SkQueueType::AIV, SkQueueType::AIV}));
}

TEST_F(SkQueueTypeResolverTest, ClearsOutputOnFailure) {
  std::vector<SkQueueType> queues{SkQueueType::AIC};

  EXPECT_FALSE(SkQueueTypeResolver(graph).Resolve({nullptr}, queues));
  EXPECT_TRUE(queues.empty());
}
