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

TEST_F(AotSystemTest, ScopeMarkersKeepStreamNameAndLaunchOrder) {
  sk::test::Model model;
  auto stream = model.AddStream();
  ASSERT_EQ(aclskScopeBegin("scope", stream), ACL_SUCCESS);
  ASSERT_EQ(aclskScopeEnd("scope", stream), ACL_SUCCESS);
  const auto calls = model.Launches();
  ASSERT_EQ(calls.size(), 6U);
  const std::vector<std::string> expected{"sk_scope_kernel_begin_dav_2201", "sk_placeholder_kernel_dav_2201",
                                          "sk_placeholder_kernel_dav_2201", "sk_placeholder_kernel_dav_2201",
                                          "sk_placeholder_kernel_dav_2201", "sk_scope_kernel_end_dav_2201"};
  for (size_t i = 0; i < calls.size(); ++i) {
    EXPECT_EQ(calls[i].stream, stream);
    EXPECT_EQ(calls[i].scope, "scope");
    EXPECT_EQ(calls[i].function, expected[i]);
  }
  EXPECT_EQ(model.Tasks(stream).size(), 6U);
}

TEST_F(AotSystemTest, EmptyScopeNameDoesNotDispatchKernel) {
  sk::test::Model model;
  auto stream = model.AddStream();
  EXPECT_EQ(aclskScopeBegin("", stream), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(aclskScopeEnd("", stream), ACL_ERROR_INVALID_PARAM);
  EXPECT_TRUE(model.Launches().empty());
  EXPECT_TRUE(model.Tasks(stream).empty());
}

TEST_F(AotSystemTest, ScopeMarkerCopyFailuresPreserveAlreadyDispatchedPrefix) {
  for (bool begin : {false, true}) {
    for (int failOnCall : {1, 2, 3}) {
      SCOPED_TRACE(begin);
      SCOPED_TRACE(failOnCall);
      SkUtResetTestControls();
      sk::test::Model model;
      const auto stream = model.AddStream();
      SkUtSetSecurecMemcpyFailOnCall(failOnCall);
      const auto result = begin ? aclskScopeBegin("copy_failure", stream) : aclskScopeEnd("copy_failure", stream);
      EXPECT_EQ(result, ACL_ERROR_INVALID_PARAM);
      const auto launches = model.Launches();
      ASSERT_EQ(launches.size(), static_cast<size_t>(failOnCall - 1));
      for (size_t i = 0; i < launches.size(); ++i) {
        EXPECT_EQ(launches[i].scope, "copy_failure");
        EXPECT_EQ(launches[i].stream, stream);
        EXPECT_EQ(launches[i].function,
                  begin && i == 0 ? "sk_scope_kernel_begin_dav_2201" : "sk_placeholder_kernel_dav_2201");
      }
    }
  }
}

TEST_F(AotSystemTest, OversizedScopeNameFailsWithoutDispatchingMarkers) {
  sk::test::Model model;
  const auto stream = model.AddStream();
  const std::string name(4096, 'x');
  EXPECT_EQ(aclskScopeBegin(name.c_str(), stream), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(aclskScopeEnd(name.c_str(), stream), ACL_ERROR_INVALID_PARAM);
  EXPECT_TRUE(model.Launches().empty());
}
