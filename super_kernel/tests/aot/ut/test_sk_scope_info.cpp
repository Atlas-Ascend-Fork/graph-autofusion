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
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>
#include "sk_scope_info.h"

class ScopeIdGeneratorTest : public testing::Test {};

TEST_F(ScopeIdGeneratorTest, NextId_IncrementSequentially) {
  uint16_t id0 = ScopeIdGenerator::Instance().NextId();
  uint16_t id1 = ScopeIdGenerator::Instance().NextId();
  uint16_t id2 = ScopeIdGenerator::Instance().NextId();

  EXPECT_EQ(id1, id0 + 1);
  EXPECT_EQ(id2, id1 + 1);
}

TEST_F(ScopeIdGeneratorTest, NextId_ConcurrentCallsReturnUniqueIds) {
  constexpr uint32_t threadCount = 32;
  constexpr uint32_t idsPerThread = 1024;
  std::atomic<uint32_t> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::vector<uint16_t>> ids(threadCount);
  std::vector<std::thread> threads;

  for (uint32_t threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
    threads.emplace_back([&, threadIndex]() {
      ids[threadIndex].reserve(idsPerThread);
      ready.fetch_add(1, std::memory_order_relaxed);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (uint32_t i = 0; i < idsPerThread; ++i) {
        ids[threadIndex].push_back(ScopeIdGenerator::Instance().NextId());
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != threadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  std::vector<uint16_t> flattened;
  flattened.reserve(threadCount * idsPerThread);
  for (const auto &threadIds : ids) {
    flattened.insert(flattened.end(), threadIds.begin(), threadIds.end());
  }
  std::sort(flattened.begin(), flattened.end());
  EXPECT_EQ(std::adjacent_find(flattened.begin(), flattened.end()), flattened.end());
}

class SuperKernelScopeInfoIdTest : public testing::Test {};

TEST_F(SuperKernelScopeInfoIdTest, AddNode_BindsNodeToScope) {
  SuperKernelKernelNode node(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
  SuperKernelScopeInfo scope;

  scope.AddNode(&node);

  EXPECT_EQ(node.GetScopeId(), scope.GetScopeId());
}

TEST_F(SuperKernelScopeInfoIdTest, SetNodes_UpdatesNodeScopeMembership) {
  SuperKernelKernelNode oldNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
  SuperKernelKernelNode newNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
  SuperKernelScopeInfo scope;
  scope.AddNode(&oldNode);

  scope.SetNodes({&newNode});

  EXPECT_EQ(oldNode.GetScopeId(), INVALID_SCOPE_ID);
  EXPECT_EQ(newNode.GetScopeId(), scope.GetScopeId());
}

TEST_F(SuperKernelScopeInfoIdTest, ScopeId_AssignedAtCreation) {
  uint16_t prevId = ScopeIdGenerator::Instance().NextId();
  SuperKernelScopeInfo scope1;
  SuperKernelScopeInfo scope2;
  SuperKernelScopeInfo scope3;

  EXPECT_EQ(scope1.GetScopeId(), prevId + 1);
  EXPECT_EQ(scope2.GetScopeId(), prevId + 2);
  EXPECT_EQ(scope3.GetScopeId(), prevId + 3);
}

TEST_F(SuperKernelScopeInfoIdTest, ScopeId_NotRecycledOnDestroy) {
  uint16_t prevId = ScopeIdGenerator::Instance().NextId();

  {
    SuperKernelScopeInfo scope1;
    EXPECT_EQ(scope1.GetScopeId(), prevId + 1);
  }

  SuperKernelScopeInfo scope2;
  EXPECT_EQ(scope2.GetScopeId(), prevId + 2);
}

TEST_F(SuperKernelScopeInfoIdTest, ScopeId_MovedScopeKeepsOriginalId) {
  uint16_t prevId = ScopeIdGenerator::Instance().NextId();

  SuperKernelScopeInfo scope1;
  uint16_t originalId = scope1.GetScopeId();
  EXPECT_EQ(originalId, prevId + 1);

  SuperKernelScopeInfo scope2 = std::move(scope1);
  EXPECT_EQ(scope2.GetScopeId(), originalId);
}

TEST_F(SuperKernelScopeInfoIdTest, ScopeId_AssignmentMoveKeepsId) {
  uint16_t prevId = ScopeIdGenerator::Instance().NextId();

  SuperKernelScopeInfo scope1;
  uint16_t originalId1 = scope1.GetScopeId();

  SuperKernelScopeInfo scope2;
  uint16_t originalId2 = scope2.GetScopeId();

  scope2 = std::move(scope1);
  EXPECT_EQ(scope2.GetScopeId(), originalId1);
  EXPECT_NE(scope2.GetScopeId(), originalId2);
}

TEST_F(SuperKernelScopeInfoIdTest, ScopeId_AssignmentMoveClearsReplacedNodeMembership) {
  SuperKernelKernelNode sourceNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
  SuperKernelKernelNode replacedNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
  SuperKernelScopeInfo sourceScope;
  SuperKernelScopeInfo targetScope;
  sourceScope.AddNode(&sourceNode);
  targetScope.AddNode(&replacedNode);
  const uint16_t sourceScopeId = sourceScope.GetScopeId();

  targetScope = std::move(sourceScope);

  EXPECT_EQ(replacedNode.GetScopeId(), INVALID_SCOPE_ID);
  EXPECT_EQ(sourceNode.GetScopeId(), sourceScopeId);
  EXPECT_EQ(targetScope.GetScopeId(), sourceScopeId);
}

TEST_F(SuperKernelScopeInfoIdTest, ScopeId_PersistAcrossSessions) {
  uint16_t prevId = ScopeIdGenerator::Instance().NextId();
  SuperKernelScopeInfo scope1;
  uint16_t id1 = scope1.GetScopeId();
  EXPECT_EQ(id1, prevId + 1);

  SuperKernelScopeInfo scope2;
  uint16_t id2 = scope2.GetScopeId();
  EXPECT_EQ(id2, id1 + 1);
}
