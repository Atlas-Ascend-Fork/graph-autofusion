/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/PlanSelector/PlanSelector.h"
#include "common/fingerprint/schedule_fingerprint.h"

#include <gtest/gtest.h>

namespace {

TEST(ScheduleSpaceBridgeTest, FingerprintAndArtifactKeyAreStable) {
  ascir::FusedScheduledResult result;
  const auto first = mlir::common::ComputeScheduleFingerprint(result);
  const auto second = mlir::common::ComputeScheduleFingerprint(result);
  EXPECT_EQ(first, second);

  mlir::common::ArtifactVersionSet versions;
  mlir::common::CompileOptions options;
  const auto key = mlir::common::BuildArtifactCacheKey(first, versions, options);
  versions.bridge_version = "v2";
  EXPECT_NE(key, mlir::common::BuildArtifactCacheKey(first, versions, options));
}

TEST(ScheduleSpaceBridgeTest, DynamicSelectionRejectsEmptyFusion) {
  ascir::FusedScheduledResult result;
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  EXPECT_FALSE(mlir::bridge::SelectDynamicCandidates(result, selections, &diagnostic));
  EXPECT_FALSE(diagnostic.empty());
}

}  // namespace
