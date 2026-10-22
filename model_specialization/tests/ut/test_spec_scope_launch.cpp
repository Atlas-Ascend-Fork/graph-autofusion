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
#include "spec_scope_launch.h"
#include "dlog_pub.h"
#include <array>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <new>
#include <stdexcept>

namespace {
int64_t currentArch = 2201;
aclError deviceResult = ACL_SUCCESS;
aclError archResult = ACL_SUCCESS;
size_t deviceQueryCount = 0;
size_t archQueryCount = 0;
constexpr int32_t CURRENT_DEVICE = 3;
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

aclError GetDevice(int32_t *deviceId) {
  ++deviceQueryCount;
  *deviceId = CURRENT_DEVICE;
  return deviceResult;
}

aclError GetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value) {
  EXPECT_EQ(deviceId, CURRENT_DEVICE);
  EXPECT_EQ(attr, ACL_DEV_ATTR_NPU_ARCH);
  ++archQueryCount;
  *value = currentArch;
  return archResult;
}

class SpecScopeLaunchTest : public testing::Test {
 protected:
  void SetUp() override {
    currentArch = 2201;
    deviceResult = archResult = ACL_SUCCESS;
    deviceQueryCount = archQueryCount = 0;
    MOCKER(aclrtGetDevice).stubs().will(invoke(GetDevice));
    MOCKER(aclrtGetDeviceInfo).stubs().will(invoke(GetDeviceInfo));
    launchCounts.fill(0);
    exceptionKind = 0;
    ut_log::LogBuffer::Instance().Clear();
  }
  void TearDown() override {
    GlobalMockObject::verify();
  }
};
}  // namespace

extern "C" void spec_scope_kernel_begin_do_dav_2201(void *stream) {
  RaiseException();
  EXPECT_EQ(stream, expectedStream);
  ++launchCounts[0];
}
extern "C" void spec_scope_kernel_end_do_dav_2201(void *stream) {
  RaiseException();
  EXPECT_EQ(stream, expectedStream);
  ++launchCounts[1];
}
extern "C" void spec_scope_kernel_begin_do_dav_3510(void *stream) {
  RaiseException();
  EXPECT_EQ(stream, expectedStream);
  ++launchCounts[2];
}
extern "C" void spec_scope_kernel_end_do_dav_3510(void *stream) {
  RaiseException();
  EXPECT_EQ(stream, expectedStream);
  ++launchCounts[3];
}

TEST_F(SpecScopeLaunchTest, DispatchesOneMarkerPerCallForEachArchWithoutCaptureQuery) {
  for (int64_t arch : {2201, 3510}) {
    currentArch = arch;
    EXPECT_EQ(aclmdlRISpecScopeBegin(expectedStream), ACL_SUCCESS);
    EXPECT_EQ(aclmdlRISpecScopeEnd(expectedStream), ACL_SUCCESS);
  }
  EXPECT_EQ(deviceQueryCount, 4U);
  EXPECT_EQ(archQueryCount, 4U);
  EXPECT_EQ(launchCounts, (std::array<size_t, 4>{1, 1, 1, 1}));
}
TEST_F(SpecScopeLaunchTest, RejectedRequestNeverLaunchesMarker) {
  struct FailureCase {
    aclrtStream stream;
    aclError expected;
    int64_t arch;
    const char *message;
    aclError deviceError = ACL_SUCCESS;
    aclError archError = ACL_SUCCESS;
  };
  const std::array<FailureCase, 4> cases{
      {{nullptr, ACL_ERROR_INVALID_PARAM, 2201, "Invalid Spec scope marker request"},
       {expectedStream, ACL_ERROR_FEATURE_UNSUPPORTED, 0, "Unsupported Spec scope marker arch"},
       {expectedStream, ACL_ERROR_RT_INSUFFICIENT_INPUT_ARRAY, 2201, "Spec scope current device query failed",
        ACL_ERROR_RT_INSUFFICIENT_INPUT_ARRAY},
       {expectedStream, ACL_ERROR_RT_INSUFFICIENT_INPUT_ARRAY, 2201, "Spec scope architecture query failed",
        ACL_SUCCESS, ACL_ERROR_RT_INSUFFICIENT_INPUT_ARRAY}}};
  for (const auto &entry : cases) {
    SCOPED_TRACE(entry.message);
    currentArch = entry.arch;
    deviceResult = entry.deviceError;
    archResult = entry.archError;
    deviceQueryCount = archQueryCount = 0;
    ut_log::LogBuffer::Instance().Clear();
    EXPECT_EQ(aclmdlRISpecScopeBegin(entry.stream), entry.expected);
    EXPECT_EQ(aclmdlRISpecScopeEnd(entry.stream), entry.expected);
    EXPECT_EQ(deviceQueryCount, entry.stream != nullptr ? 2U : 0U);
    EXPECT_EQ(archQueryCount, entry.stream != nullptr && entry.deviceError == ACL_SUCCESS ? 2U : 0U);
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
