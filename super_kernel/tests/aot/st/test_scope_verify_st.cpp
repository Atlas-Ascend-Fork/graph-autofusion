/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "st_fixture.h"
#include <limits>

namespace {
aclskScopeVerifyNodeInfo Compute(int64_t id, aclskScopeVerifyKernelType type, uint32_t blocks, uint32_t cube,
                                 uint32_t vector, int32_t scheMode) {
  aclskScopeVerifyNodeInfo node{};
  node.taskId = id;
  node.scopeId = 1;
  node.taskType = ACLSK_SCOPE_VERIFY_NODE_COMPUTE;
  node.kernelType = type;
  node.numBlocks = blocks;
  node.taskRatio[0] = cube;
  node.taskRatio[1] = vector;
  node.scheMode = scheMode;
  return node;
}
}  // namespace
TEST_F(AotSystemTest, VerifyCompatibleComputeGraphNeedsNoSplit) {
  std::vector<aclskScopeVerifyNodeInfo> nodes{Compute(1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 4, 1, 0, 0),
                                              Compute(2, ACLSK_SCOPE_VERIFY_KERNEL_VECTOR, 4, 0, 1, 0)};
  aclskScopeVerifyGraphInfo graph{nodes.data(), nodes.size(), 0, nullptr};
  size_t count = 123;
  ASSERT_EQ(aclskScopeVerify(&graph, 0, nullptr, &count), ACL_SUCCESS);
  EXPECT_EQ(count, 0U);
}
TEST_F(AotSystemTest, VerifyScheModeSplitReferencesOriginalInputNode) {
  std::vector<aclskScopeVerifyNodeInfo> nodes{Compute(1, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 1, 2, 1),
                                              Compute(2, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 2, 1, 0, 1)};
  aclskScopeVerifyGraphInfo graph{nodes.data(), nodes.size(), 0, nullptr};
  aclskScopeVerifySplitResult result{};
  size_t count = 0;
  ASSERT_EQ(aclskScopeVerify(&graph, 1, &result, &count), ACL_SUCCESS);
  ASSERT_EQ(count, 1U);
  EXPECT_EQ(result.splitNode, &nodes[1]);
  EXPECT_EQ(result.splitType, ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE);
  EXPECT_EQ(result.splitReason, ACLSK_SCOPE_VERIFY_SYNCALL_OP_DROP);
  EXPECT_EQ(result.extendType, 0);
  EXPECT_EQ(result.extendInfo, nullptr);
}

TEST_F(AotSystemTest, VerifyReportsRequiredCapacityWithoutOverwritingCallerBuffer) {
  std::vector<aclskScopeVerifyNodeInfo> nodes{Compute(1, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 1, 2, 1),
                                              Compute(2, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 2, 1, 0, 1),
                                              Compute(3, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 1, 1, 0, 1)};
  aclskScopeVerifyGraphInfo graph{nodes.data(), nodes.size(), 0, nullptr};
  aclskScopeVerifySplitResult result{};
  result.splitNode = &nodes[0];
  result.extendType = 73;
  size_t count = 0;
  EXPECT_EQ(aclskScopeVerify(&graph, 1, &result, &count), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(count, 2U);
  EXPECT_EQ(result.splitNode, &nodes[0]);
  EXPECT_EQ(result.extendType, 73);
  std::vector<aclskScopeVerifySplitResult> results(count);
  ASSERT_EQ(aclskScopeVerify(&graph, results.size(), results.data(), &count), ACL_SUCCESS);
  ASSERT_EQ(count, 2U);
  EXPECT_EQ(results[0].splitNode, &nodes[1]);
  EXPECT_EQ(results[1].splitNode, &nodes[2]);
}

TEST_F(AotSystemTest, VerifyDynamicCoreLimitsChangeScheModeSplitDecision) {
  std::vector<aclskScopeVerifyNodeInfo> nodes{Compute(1, ACLSK_SCOPE_VERIFY_KERNEL_MIX, 4, 1, 2, 1),
                                              Compute(2, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 2, 1, 0, 1)};
  nodes[1].flag = 1;
  nodes[1].coreLimit[0] = 4;
  aclskScopeVerifyGraphInfo graph{nodes.data(), nodes.size(), 0, nullptr};
  size_t count = 0;
  ASSERT_EQ(aclskScopeVerify(&graph, 0, nullptr, &count), ACL_SUCCESS);
  EXPECT_EQ(count, 0U);
  nodes[1].flag = 0;
  aclskScopeVerifySplitResult result{};
  ASSERT_EQ(aclskScopeVerify(&graph, 1, &result, &count), ACL_SUCCESS);
  EXPECT_EQ(count, 1U);
  EXPECT_EQ(result.splitNode, &nodes[1]);
}

TEST_F(AotSystemTest, VerifyCrossStreamDeadlockExcludesWaitNode) {
  std::vector<aclskScopeVerifyNodeInfo> nodes{Compute(1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 20, 1, 0, 0),
                                              {},
                                              Compute(3, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 20, 1, 0, 0),
                                              Compute(4, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 20, 1, 0, 0),
                                              {}};
  nodes[1].taskId = 2;
  nodes[1].taskType = ACLSK_SCOPE_VERIFY_NODE_WAIT;
  nodes[1].eventId = 100;
  nodes[1].scopeId = 1;
  nodes[3].streamId = 1;
  nodes[3].scopeId = 2;
  nodes[4].taskId = 5;
  nodes[4].taskType = ACLSK_SCOPE_VERIFY_NODE_NOTIFY;
  nodes[4].eventId = 100;
  nodes[4].streamId = 1;
  nodes[4].scopeId = 2;
  aclskScopeVerifyGraphInfo graph{nodes.data(), nodes.size(), 0, nullptr};
  aclskScopeVerifySplitResult results[2]{};
  size_t count = 0;
  ASSERT_EQ(aclskScopeVerify(&graph, 2, results, &count), ACL_SUCCESS);
  ASSERT_EQ(count, 1U);
  EXPECT_EQ(results[0].splitNode, &nodes[1]);
  EXPECT_EQ(results[0].splitType, ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE);
  EXPECT_EQ(results[0].splitReason, ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED);
}

TEST_F(AotSystemTest, VerifyRejectsInvalidGraphAndNodeDescriptionsAndClearsCount) {
  size_t count = 99;
  EXPECT_EQ(aclskScopeVerify(nullptr, 0, nullptr, &count), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(count, 0U);
  auto valid = Compute(1, ACLSK_SCOPE_VERIFY_KERNEL_CUBE, 4, 1, 0, 0);
  std::vector<aclskScopeVerifyNodeInfo> invalid(7, valid);
  invalid[0].taskId = -1;
  invalid[1].streamId = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  invalid[2].extendType = 1;
  invalid[3].kernelType = ACLSK_SCOPE_VERIFY_KERNEL_NO_AICORE;
  invalid[4].flag = 1;
  invalid[4].coreLimit[0] = -1;
  invalid[5].taskType = ACLSK_SCOPE_VERIFY_NODE_WAIT;
  invalid[5].eventId = -1;
  invalid[6].kernelType = ACLSK_SCOPE_VERIFY_KERNEL_MIX;
  invalid[6].taskRatio[1] = 3;
  for (size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    aclskScopeVerifyGraphInfo graph{&invalid[i], 1, 0, nullptr};
    count = 99;
    EXPECT_EQ(aclskScopeVerify(&graph, 0, nullptr, &count), ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(count, 0U);
  }
  aclskScopeVerifyNodeInfo duplicates[]{valid, valid};
  aclskScopeVerifyGraphInfo graph{duplicates, 2, 0, nullptr};
  EXPECT_EQ(aclskScopeVerify(&graph, 0, nullptr, &count), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(count, 0U);
}
