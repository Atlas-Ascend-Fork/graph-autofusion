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
TEST_F(SpecScopeLaunchTest, RejectedCaptureOrRequestNeverLaunchesMarker) {
  struct FailureCase {
    aclrtStream stream;
    aclmdlRICaptureStatus status;
    aclError query, expected;
    SkKernelArch arch = SkKernelArch::DAV_2201;
    const char *message;
  };
  const std::array<FailureCase, 6> cases{
      {{nullptr, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE, ACL_SUCCESS, ACL_ERROR_INVALID_PARAM, SkKernelArch::DAV_2201,
        "Invalid Spec scope marker request"},
       {expectedStream, ACL_MODEL_RI_CAPTURE_STATUS_NONE, ACL_SUCCESS, ACL_ERROR_RT_STREAM_NOT_CAPTURED,
        SkKernelArch::DAV_2201, "Spec scope stream is not captured"},
       {expectedStream, ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED, ACL_SUCCESS, ACL_ERROR_RT_STREAM_CAPTURE_INVALIDATED,
        SkKernelArch::DAV_2201, "Spec scope capture is invalidated"},
       {expectedStream, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE, ACL_ERROR_FAILURE, ACL_ERROR_FAILURE,
        SkKernelArch::DAV_2201, "Spec scope capture query failed"},
       {expectedStream, static_cast<aclmdlRICaptureStatus>(99), ACL_SUCCESS, ACL_ERROR_FAILURE, SkKernelArch::DAV_2201,
        "Unknown Spec scope capture status"},
       {expectedStream, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE, ACL_SUCCESS, ACL_ERROR_FEATURE_UNSUPPORTED,
        SkKernelArch::UNKNOWN, "Unsupported Spec scope marker arch"}}};
  for (const auto &entry : cases) {
    SCOPED_TRACE(entry.message);
    captureStatus = entry.status;
    queryResult = entry.query;
    currentArch = entry.arch;
    queryCount = 0;
    ut_log::LogBuffer::Instance().Clear();
    EXPECT_EQ(aclmdlRISpecScopeBegin(entry.stream), entry.expected);
    EXPECT_EQ(aclmdlRISpecScopeEnd(entry.stream), entry.expected);
    EXPECT_EQ(queryCount, entry.stream == nullptr ? 0U : 2U);
    EXPECT_EQ(launchCounts, (std::array<size_t, 4>{}));
    EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(entry.message), std::string::npos);
  }
  EXPECT_EQ(model_spec::LaunchSpecScopeMarker(expectedStream, static_cast<model_spec::SpecScopeMarkerKind>(99)),
            ACL_ERROR_INVALID_PARAM);
}

TEST_F(SpecScopeLaunchTest, PublicEntryPointsMapAllExceptionKinds) {
  const std::array<aclError, 5> errors{ACL_ERROR_BAD_ALLOC, ACL_ERROR_BAD_ALLOC, ACL_ERROR_INVALID_PARAM,
                                       ACL_ERROR_FAILURE, ACL_ERROR_FAILURE};
  for (size_t index = 0; index < errors.size(); ++index) {
    exceptionKind = static_cast<int>(index + 1);
    EXPECT_EQ(aclmdlRISpecScopeBegin(expectedStream), errors[index]);
    EXPECT_EQ(aclmdlRISpecScopeEnd(expectedStream), errors[index]);
  }
  EXPECT_EQ(launchCounts, (std::array<size_t, 4>{}));
}
