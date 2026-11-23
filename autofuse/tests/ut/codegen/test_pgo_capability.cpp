/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/pgo/pgo_capability.h"

#include <gtest/gtest.h>

namespace codegen::pgo {
namespace {

TEST(PgoCapability, RejectsNullOutput) {
  EXPECT_EQ(ProbePgoCapability(reinterpret_cast<void *>(1), nullptr), PGO_INVALID_ARGUMENT);
}

TEST(PgoCapability, RejectsNullStream) {
  PgoCapability capability;
  EXPECT_EQ(ProbePgoCapability(nullptr, &capability), PGO_CAPABILITY_UNAVAILABLE);
  EXPECT_FALSE(capability.stars_task_track_supported);
  EXPECT_FALSE(capability.device_timestamp_supported);
}

TEST(PgoCapability, DefaultProbeNeverClaimsSupportWithoutDevice) {
  PgoCapability capability;
  const int status = ProbePgoCapability(&capability);
  if (status != PGO_SUCCESS) {
    EXPECT_FALSE(capability.stars_task_track_supported);
    EXPECT_FALSE(capability.device_timestamp_supported);
  }
}

TEST(PgoCapability, RequiresExclusiveReporterOwnership) {
  PgoCapability capability;
  EXPECT_EQ(ProbePgoCapability(PgoReporterOwnership::kUnknown, reinterpret_cast<void *>(1), &capability),
            PGO_REPORTER_BUSY);
  EXPECT_EQ(ProbePgoCapability(PgoReporterOwnership::kExternal, reinterpret_cast<void *>(1), &capability),
            PGO_REPORTER_BUSY);
}

TEST(PgoCapability, ActivityReporterIsNeverInferredFromRuntimeSupport) {
  PgoCapability capability;
  const int status = ProbePgoCapability(&capability);
  if (status != PGO_SUCCESS) {
    EXPECT_FALSE(capability.kernel_activity_reporter_supported);
    EXPECT_EQ(capability.profile_status, PGO_PROFILE_UNSUPPORTED);
  }
}

TEST(PgoCapability, PlatformControlsMsptiFrequencyAndSamplePeriod) {
  EXPECT_EQ(ResolvePgoDeviceTicksPerMs(PGO_DATA_UNAVAILABLE, 0, PgoPlatformType::kAscend910B), 50U);
  EXPECT_EQ(ResolvePgoDeviceTicksPerMs(PGO_DATA_UNAVAILABLE, 0, PgoPlatformType::kAscend310B), 50U);
  EXPECT_EQ(ResolvePgoDeviceTicksPerMs(PGO_DATA_UNAVAILABLE, 0, PgoPlatformType::kAscendV6), 1000U);
  EXPECT_EQ(ResolvePgoDeviceTicksPerMs(PGO_SUCCESS, 123, PgoPlatformType::kAscendV6), 123U);
  EXPECT_EQ(ResolvePgoDeviceTicksPerMs(PGO_SUCCESS, 0, PgoPlatformType::kAscendV6), 0U);
  EXPECT_EQ(ResolvePgoStarsSamplePeriod(PgoPlatformType::kAscend910B), 20U);
  EXPECT_EQ(ResolvePgoStarsSamplePeriod(PgoPlatformType::kAscend310B), 20U);
  EXPECT_EQ(ResolvePgoStarsSamplePeriod(PgoPlatformType::kAscendV6), 0U);
  EXPECT_EQ(ResolvePgoStarsSamplePeriod(PgoPlatformType::kUnknown), 20U);
}

TEST(PgoCapability, DistinguishesUnsupportedProfileFromUnavailableCapability) {
  EXPECT_EQ(ResolvePgoCapabilityStatus(true, true, true, PgoPlatformType::kAscend910B), PGO_SUCCESS);
  EXPECT_EQ(ResolvePgoCapabilityStatus(true, true, false, PgoPlatformType::kAscend910B), PGO_PROFILE_UNSUPPORTED);
  EXPECT_EQ(ResolvePgoCapabilityStatus(true, true, true, PgoPlatformType::kUnknown), PGO_PROFILE_UNSUPPORTED);
  EXPECT_EQ(ResolvePgoCapabilityStatus(false, true, true, PgoPlatformType::kAscend910B), PGO_CAPABILITY_UNAVAILABLE);
  EXPECT_EQ(ResolvePgoCapabilityStatus(true, false, true, PgoPlatformType::kAscend910B), PGO_CAPABILITY_UNAVAILABLE);
}

}  // namespace
}  // namespace codegen::pgo
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
