/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdlib>
#include <map>
#include <optional>
#include <gtest/gtest.h>
#include "super_kernel.h"
#include "model_fixture.h"
#include "ut_common_stubs.h"
#include "dlog_pub.h"

class AotSystemTest : public testing::Test {
 protected:
  void SetUp() override {
    for (const char *name : {"ASCEND_OP_COMPILE_SAVE_KERNEL_META", "ASCEND_PROF_SK_ON"}) {
      const char *value = std::getenv(name);
      env_[name] = value == nullptr ? std::nullopt : std::optional<std::string>(value);
      unsetenv(name);
    }
    SkUtResetTestControls();
    ut_log::LogBuffer::Instance().Clear();
    ASSERT_EQ(sk::test::OutstandingAllocations(), 0U);
  }
  void TearDown() override {
    // Model locals are destroyed before this method: assert before reset,
    // so clearing controls cannot hide a lost destruction callback.
    EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 0U);
    EXPECT_EQ(sk::test::OutstandingAllocations(), 0U);
    if (HasFailure()) {
      ut_log::LogBuffer::Instance().Flush();
    } else {
      ut_log::LogBuffer::Instance().Clear();
    }
    SkUtResetTestControls();
    for (const auto &[name, value] : env_) {
      if (value) {
        setenv(name.c_str(), value->c_str(), 1);
      } else {
        unsetenv(name.c_str());
      }
    }
  }

 private:
  std::map<std::string, std::optional<std::string>> env_;
};
