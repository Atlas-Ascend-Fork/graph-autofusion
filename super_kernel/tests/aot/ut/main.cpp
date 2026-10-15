/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "dlog_pub.h"

// 转发监听器基类
class ForwardingListener : public testing::TestEventListener {
 public:
  explicit ForwardingListener(testing::TestEventListener *l) : listener_(l) {}
  void OnTestProgramStart(const testing::UnitTest &u) override {
    if (listener_) listener_->OnTestProgramStart(u);
  }
  void OnTestIterationStart(const testing::UnitTest &u, int i) override {
    if (listener_) listener_->OnTestIterationStart(u, i);
  }
  void OnEnvironmentsSetUpStart(const testing::UnitTest &u) override {
    if (listener_) listener_->OnEnvironmentsSetUpStart(u);
  }
  void OnEnvironmentsSetUpEnd(const testing::UnitTest &u) override {
    if (listener_) listener_->OnEnvironmentsSetUpEnd(u);
  }
  void OnTestSuiteStart(const testing::TestSuite &s) override {
    if (listener_) listener_->OnTestSuiteStart(s);
  }
  void OnTestStart(const testing::TestInfo &t) override {
    if (listener_) listener_->OnTestStart(t);
  }
  void OnTestPartResult(const testing::TestPartResult &r) override {
    if (listener_) listener_->OnTestPartResult(r);
  }
  void OnTestEnd(const testing::TestInfo &t) override {
    if (listener_) listener_->OnTestEnd(t);
  }
  void OnTestSuiteEnd(const testing::TestSuite &s) override {
    if (listener_) listener_->OnTestSuiteEnd(s);
  }
  void OnEnvironmentsTearDownStart(const testing::UnitTest &u) override {
    if (listener_) listener_->OnEnvironmentsTearDownStart(u);
  }
  void OnEnvironmentsTearDownEnd(const testing::UnitTest &u) override {
    if (listener_) listener_->OnEnvironmentsTearDownEnd(u);
  }
  void OnTestIterationEnd(const testing::UnitTest &u, int i) override {
    if (listener_) listener_->OnTestIterationEnd(u, i);
  }
  void OnTestProgramEnd(const testing::UnitTest &u) override {
    if (listener_) listener_->OnTestProgramEnd(u);
  }

 protected:
  testing::TestEventListener *listener_;
};

// 测试失败时打印日志的监听器
class FailureLogListener : public ForwardingListener {
 public:
  using ForwardingListener::ForwardingListener;

  void OnTestStart(const testing::TestInfo &t) override {
    ut_log::LogBuffer::Instance().Clear();
    ForwardingListener::OnTestStart(t);
  }

  void OnTestEnd(const testing::TestInfo &t) override {
    if (t.result()->Failed())
      ut_log::LogBuffer::Instance().Flush();
    else
      ut_log::LogBuffer::Instance().Clear();
    ForwardingListener::OnTestEnd(t);
  }
};

int32_t main(int32_t argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  auto &listeners = testing::UnitTest::GetInstance()->listeners();
  listeners.Append(new FailureLogListener(listeners.Release(listeners.default_result_printer())));
  return RUN_ALL_TESTS();
}
