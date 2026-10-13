/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclmodel_specialization.h"
#include "sk_common.h"
#include "spec_scope_launch.h"
#include "dlog_pub.h"
#include <array>
#include <gtest/gtest.h>
#include <new>
#include <stdexcept>

namespace {
aclmdlRICaptureStatus captureStatus = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
aclError queryResult = ACL_SUCCESS;
SkKernelArch currentArch = SkKernelArch::DAV_2201;
size_t queryCount = 0;
std::array<size_t, 4> launchCounts{};
aclrtStream expectedStream = reinterpret_cast<aclrtStream>(0x100);
int exceptionKind = 0;

void RaiseException() {
  switch (exceptionKind) {
    case 1:
      throw std::bad_alloc();
    case 2:
      throw std::length_error("test");
    case 3:
      throw std::invalid_argument("test");
    case 4:
      throw std::runtime_error("test");
    case 5:
      throw 1;
    default:
      break;
  }
}

class SpecScopeLaunchTest : public testing::Test {
 protected:
  void SetUp() override {
    captureStatus = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    queryResult = ACL_SUCCESS;
    currentArch = SkKernelArch::DAV_2201;
    queryCount = 0;
    launchCounts.fill(0);
    exceptionKind = 0;
    ut_log::LogBuffer::Instance().Clear();
  }
  void CheckBoth(aclError expected, aclrtStream stream = expectedStream) {
    const size_t queriesBefore = queryCount;
    EXPECT_EQ(aclmdlRISpecScopeBegin(stream), expected);
    EXPECT_EQ(aclmdlRISpecScopeEnd(stream), expected);
    EXPECT_EQ(launchCounts, (std::array<size_t, 4>{}));
    EXPECT_EQ(queryCount - queriesBefore, stream == nullptr ? 0U : 2U);
  }
};
}  // namespace

void InitSkRuntimeConfig() {}
SkKernelArch GetCurrentSkKernelArch() {
  return currentArch;
}
extern "C" aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *model) {
  EXPECT_EQ(stream, expectedStream);
  EXPECT_NE(status, nullptr);
  EXPECT_NE(model, nullptr);
  ++queryCount;
  RaiseException();
  *status = captureStatus;
  *model = reinterpret_cast<aclmdlRI>(0x200);
  return queryResult;
}

extern "C" void spec_scope_kernel_begin_do_dav_2201(void *stream) {
  EXPECT_EQ(stream, expectedStream);
  ++launchCounts[0];
}
extern "C" void spec_scope_kernel_end_do_dav_2201(void *stream) {
  EXPECT_EQ(stream, expectedStream);
  ++launchCounts[1];
}
extern "C" void spec_scope_kernel_begin_do_dav_3510(void *stream) {
  EXPECT_EQ(stream, expectedStream);
  ++launchCounts[2];
}
extern "C" void spec_scope_kernel_end_do_dav_3510(void *stream) {
  EXPECT_EQ(stream, expectedStream);
  ++launchCounts[3];
}

TEST_F(SpecScopeLaunchTest, ActiveCaptureDispatchesOneMarkerPerCallForEachArch) {
  for (auto arch : {SkKernelArch::DAV_2201, SkKernelArch::DAV_3510}) {
    currentArch = arch;
    EXPECT_EQ(aclmdlRISpecScopeBegin(expectedStream), ACL_SUCCESS);
    EXPECT_EQ(aclmdlRISpecScopeEnd(expectedStream), ACL_SUCCESS);
  }
  EXPECT_EQ(queryCount, 4);
  EXPECT_EQ(launchCounts, (std::array<size_t, 4>{1, 1, 1, 1}));
}
TEST_F(SpecScopeLaunchTest, NullStreamDoesNotQueryOrLaunch) {
  CheckBoth(ACL_ERROR_INVALID_PARAM, nullptr);
  EXPECT_EQ(queryCount, 0);
}
TEST_F(SpecScopeLaunchTest, NoneCaptureDoesNotLaunch) {
  captureStatus = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  CheckBoth(ACL_ERROR_RT_STREAM_NOT_CAPTURED);
  const std::string logs = ut_log::LogBuffer::Instance().GetContent();
  EXPECT_NE(logs.find("[ERROR] [SK][LaunchSpecScopeMarker] Spec scope stream is not captured: markerType=Begin, "
                      "stream=0x100, status=0"),
            std::string::npos);
  EXPECT_NE(logs.find("[ERROR] [SK][LaunchSpecScopeMarker] Spec scope stream is not captured: markerType=End, "
                      "stream=0x100, status=0"),
            std::string::npos);
  EXPECT_EQ(queryCount, 2);
}
TEST_F(SpecScopeLaunchTest, InvalidatedCaptureDoesNotLaunch) {
  captureStatus = ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED;
  CheckBoth(ACL_ERROR_RT_STREAM_CAPTURE_INVALIDATED);
  const std::string logs = ut_log::LogBuffer::Instance().GetContent();
  EXPECT_NE(logs.find("[ERROR] [SK][LaunchSpecScopeMarker] Spec scope capture is invalidated: markerType=Begin, "
                      "stream=0x100, status=2"),
            std::string::npos);
  EXPECT_NE(logs.find("[ERROR] [SK][LaunchSpecScopeMarker] Spec scope capture is invalidated: markerType=End, "
                      "stream=0x100, status=2"),
            std::string::npos);
}
TEST_F(SpecScopeLaunchTest, QueryFailurePreservesErrorAndDoesNotLaunch) {
  queryResult = 123456;
  CheckBoth(queryResult);
}
TEST_F(SpecScopeLaunchTest, UnknownCaptureStatusDoesNotLaunch) {
  captureStatus = static_cast<aclmdlRICaptureStatus>(99);
  CheckBoth(ACL_ERROR_FAILURE);
}
TEST_F(SpecScopeLaunchTest, UnsupportedArchDoesNotLaunch) {
  currentArch = SkKernelArch::UNKNOWN;
  CheckBoth(ACL_ERROR_FEATURE_UNSUPPORTED);
}
TEST_F(SpecScopeLaunchTest, PublicEntryPointsMapAllExceptionKinds) {
  const std::array<aclError, 5> errors = {ACL_ERROR_BAD_ALLOC, ACL_ERROR_BAD_ALLOC, ACL_ERROR_INVALID_PARAM,
                                          ACL_ERROR_FAILURE, ACL_ERROR_FAILURE};
  for (size_t index = 0; index < errors.size(); ++index) {
    exceptionKind = static_cast<int>(index + 1);
    CheckBoth(errors[index]);
  }
}

TEST_F(SpecScopeLaunchTest, InvalidInternalMarkerKindDoesNotQueryOrLaunch) {
  EXPECT_EQ(model_spec::LaunchSpecScopeMarker(expectedStream, static_cast<model_spec::SpecScopeMarkerKind>(99)),
            ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(queryCount, 0);
  EXPECT_EQ(launchCounts, (std::array<size_t, 4>{}));
}
