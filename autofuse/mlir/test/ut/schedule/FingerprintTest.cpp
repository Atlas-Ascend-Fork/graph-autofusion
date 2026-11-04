/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/fingerprint/schedule_fingerprint.h"

#include <gtest/gtest.h>

#include "ScheduleTestUtils.h"

namespace {

TEST(FingerprintTest, IsStableForEquivalentSchedule) {
  const auto first = mlir::common::ComputeScheduleFingerprint(mlir::test::MakeFusedResult(1U, 2U, false, true));
  const auto second = mlir::common::ComputeScheduleFingerprint(mlir::test::MakeFusedResult(1U, 2U, false, true));
  EXPECT_EQ(first, second);
  EXPECT_EQ(first.value.size(), 32U);
}

TEST(FingerprintTest, ChangesWhenScheduleStructureChanges) {
  const auto one_template = mlir::common::ComputeScheduleFingerprint(mlir::test::MakeFusedResult(1U, 1U));
  const auto two_templates = mlir::common::ComputeScheduleFingerprint(mlir::test::MakeFusedResult(1U, 2U));
  EXPECT_NE(one_template, two_templates);
}

// Regression: constant size vars must not affect the fingerprint, matching
// tiling_field_schema.cpp's AppendGraphFields, which also skips
// IsConstExpr() size vars. Two schedules that differ only in which constant
// value a compile-time-known size var happens to hold are the same schedule
// structurally -- they must hash identically, otherwise every recompile of
// the very same kernel with a different constant would needlessly miss the
// artifact cache.
TEST(FingerprintTest, IsUnaffectedByConstantSizeVarValue) {
  const auto with_seven = mlir::common::ComputeScheduleFingerprint(mlir::test::MakeFusedResultWithConstSizeVar(7));
  const auto with_ninety_nine =
      mlir::common::ComputeScheduleFingerprint(mlir::test::MakeFusedResultWithConstSizeVar(99));
  EXPECT_EQ(with_seven, with_ninety_nine);
}

TEST(FingerprintTest, CacheKeyIncludesVersionsAndOptions) {
  const auto fingerprint = mlir::common::ComputeScheduleFingerprint(mlir::test::MakeFusedResult());
  mlir::common::ArtifactVersionSet versions;
  mlir::common::CompileOptions options;
  const auto base = mlir::common::BuildArtifactCacheKey(fingerprint, versions, options);

  versions.bridge_version = "bridge-v2";
  EXPECT_NE(base, mlir::common::BuildArtifactCacheKey(fingerprint, versions, options));
  versions.bridge_version.clear();
  options.values.emplace("soc", "Ascend910B");
  EXPECT_NE(base, mlir::common::BuildArtifactCacheKey(fingerprint, versions, options));
}

// Regression: a dynamic-mode artifact and a static-mode one compiled for a
// concrete shape must not share a cache key, and neither may two static
// artifacts compiled for different concrete shapes -- both are represented
// by ArtifactVersionSet fields the earlier version of this struct did not
// have.
TEST(FingerprintTest, CacheKeyDistinguishesShapeModeAndStaticConstants) {
  const auto fingerprint = mlir::common::ComputeScheduleFingerprint(mlir::test::MakeFusedResult());
  mlir::common::CompileOptions options;

  mlir::common::ArtifactVersionSet dynamic_versions;
  dynamic_versions.is_dynamic_shape = true;
  const auto dynamic_key = mlir::common::BuildArtifactCacheKey(fingerprint, dynamic_versions, options);

  mlir::common::ArtifactVersionSet static_versions_a;
  static_versions_a.is_dynamic_shape = false;
  static_versions_a.static_shape_constants = {{"s0", 32}};
  const auto static_key_a = mlir::common::BuildArtifactCacheKey(fingerprint, static_versions_a, options);
  EXPECT_NE(dynamic_key, static_key_a);

  mlir::common::ArtifactVersionSet static_versions_b;
  static_versions_b.is_dynamic_shape = false;
  static_versions_b.static_shape_constants = {{"s0", 64}};
  const auto static_key_b = mlir::common::BuildArtifactCacheKey(fingerprint, static_versions_b, options);
  EXPECT_NE(static_key_a, static_key_b);
}

}  // namespace
