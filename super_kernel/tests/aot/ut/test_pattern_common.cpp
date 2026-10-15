/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common/pattern_common.h"

#include "gtest/gtest.h"

namespace sk {
namespace {

// TrimString / IsValidRegexPattern / MatchKernelNamePattern are shared SuperKernel
// pattern rules, reused by SuperKernelOptionsManager::MatchRegex and the static-normal
// compile feature manager. These are pure pattern tests, kept here with the AOT code.
TEST(PatternCommonTest, CharacterValidationAndMatchingHaveSeparateResponsibilities) {
  EXPECT_TRUE(sk::IsValidRegexPattern("*add"));
  EXPECT_FALSE(sk::MatchKernelNamePattern("*add", "add"));
  EXPECT_FALSE(sk::IsValidRegexPattern(" add.* "));
  EXPECT_EQ(sk::TrimString(" \tadd.*\n"), "add.*");
  EXPECT_TRUE(sk::MatchKernelNamePattern(" \tadd.*\n", "addcustom"));
  EXPECT_FALSE(sk::IsValidRegexPattern(""));
  EXPECT_FALSE(sk::IsValidRegexPattern("add[0]"));
  EXPECT_FALSE(sk::MatchKernelNamePattern(" \t ", "add"));
  EXPECT_FALSE(sk::MatchKernelNamePattern("add[0]", "add0"));
}

TEST(PatternCommonTest, SharedSuperKernelRuleSemantics) {
  EXPECT_TRUE(sk::MatchKernelNamePattern("Add*", "Ad"));
  EXPECT_TRUE(sk::MatchKernelNamePattern("Add*", "Adddd"));
  EXPECT_FALSE(sk::MatchKernelNamePattern("Add*", "Add_123"));
  EXPECT_TRUE(sk::MatchKernelNamePattern("a.c", "abc"));
  EXPECT_FALSE(sk::MatchKernelNamePattern("a.c", "ac"));
  EXPECT_TRUE(sk::MatchKernelNamePattern("a*", ""));
  EXPECT_TRUE(sk::MatchKernelNamePattern(" \t.*Add.*\n", "xAdd_1"));
  EXPECT_FALSE(sk::MatchKernelNamePattern("Add", "add"));
}

}  // namespace
}  // namespace sk
