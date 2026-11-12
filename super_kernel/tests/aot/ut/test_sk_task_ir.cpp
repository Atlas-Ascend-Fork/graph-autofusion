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

#include "sk_task_ir.h"

TEST(SkTaskIRTest, StoresUnifiedTasksAndResolvedQueues) {
  SkTaskDesc kernel;
  kernel.nodeId = 1;
  kernel.kind = SkTaskKind::KERNEL;
  kernel.queueType = SkQueueType::MIX_1_2;
  kernel.payload = SkKernelDesc{};

  SkTaskDesc event;
  event.nodeId = 2;
  event.kind = SkTaskKind::WAIT;
  event.queueType = SkQueueType::AIV;
  event.payload = SkEventDesc{};

  SkTaskIR input;
  input.tasks = {kernel, event};

  ASSERT_EQ(input.tasks.size(), 2U);
  EXPECT_EQ(input.tasks[0].queueType, SkQueueType::MIX_1_2);
  EXPECT_TRUE(std::holds_alternative<SkKernelDesc>(input.tasks[0].payload));
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIV);
  EXPECT_TRUE(std::holds_alternative<SkEventDesc>(input.tasks[1].payload));
}

TEST(SkTaskIRTest, TypedAttributesUseDefaultsAndOverwriteById) {
  std::vector<SkTaskAttribute> attributes;
  SetBoolAttribute(attributes, SkTaskAttributeId::KERNEL_SCHE_MODE, true);
  SetUint64Attribute(attributes, SkTaskAttributeId::SIMT_DYN_UBUF_SIZE, 4096U);
  SetBoolAttribute(attributes, SkTaskAttributeId::KERNEL_SCHE_MODE, false);

  ASSERT_EQ(attributes.size(), 2U);
  EXPECT_FALSE(GetBoolAttribute(attributes, SkTaskAttributeId::KERNEL_SCHE_MODE, true));
  EXPECT_TRUE(GetBoolAttribute(attributes, SkTaskAttributeId::KERNEL_BLOCK_DIM_SCALE_UP, true));
  ASSERT_TRUE(GetUint64Attribute(attributes, SkTaskAttributeId::SIMT_DYN_UBUF_SIZE).has_value());
  EXPECT_EQ(GetUint64Attribute(attributes, SkTaskAttributeId::SIMT_DYN_UBUF_SIZE).value(), 4096U);
}

TEST(SkTaskIRTest, RejectsDuplicateUnknownRequiredAndTypeMismatchedAttributes) {
  const auto unknownId = static_cast<SkTaskAttributeId>(0xFFFFU);
  std::vector<SkTaskAttribute> attributes = {
      {unknownId, SkTaskAttributeType::UINT64, SkTaskAttributeFlag::OPTIONAL, 1U}};
  EXPECT_TRUE(ValidateKernelAttributes(attributes));

  attributes[0].flags = SkTaskAttributeFlag::REQUIRED;
  EXPECT_FALSE(ValidateKernelAttributes(attributes));

  attributes = {
      {SkTaskAttributeId::KERNEL_SCHE_MODE, SkTaskAttributeType::BOOL, SkTaskAttributeFlag::OPTIONAL, 1U},
      {SkTaskAttributeId::KERNEL_SCHE_MODE, SkTaskAttributeType::BOOL, SkTaskAttributeFlag::OPTIONAL, 0U},
  };
  EXPECT_FALSE(ValidateKernelAttributes(attributes));

  attributes = {{SkTaskAttributeId::KERNEL_SCHE_MODE, SkTaskAttributeType::UINT64, SkTaskAttributeFlag::OPTIONAL, 1U}};
  EXPECT_FALSE(ValidateKernelAttributes(attributes));
}
