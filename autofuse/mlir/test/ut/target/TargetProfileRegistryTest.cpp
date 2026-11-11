/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/TargetProfileRegistry.h"

#include <string>

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace mlir::ascend {
namespace {

TEST(TargetProfileRegistryTest, ReturnsTheSameInstanceForOneSelection) {
  std::string diagnostic;
  llvm::raw_string_ostream stream(diagnostic);

  auto first = getTargetProfile(TargetSelection::builtinPlatform("Ascend910B1"), stream);
  ASSERT_TRUE(succeeded(first));
  ASSERT_TRUE(*first);

  auto second = getTargetProfile(TargetSelection::builtinPlatform("Ascend910B1"), stream);
  ASSERT_TRUE(succeeded(second));

  // Cached: the profile is parsed once and shared, so every session-scoped pass
  // in one run observes the same immutable object rather than reloading it.
  EXPECT_EQ(first->get(), second->get());
  EXPECT_TRUE(diagnostic.empty());
}

TEST(TargetProfileRegistryTest, DistinctSelectionsDoNotShareAnEntry) {
  std::string diagnostic;
  llvm::raw_string_ostream stream(diagnostic);

  auto b1 = getTargetProfile(TargetSelection::builtinPlatform("Ascend910B1"), stream);
  auto b2 = getTargetProfile(TargetSelection::builtinPlatform("Ascend910B2"), stream);
  ASSERT_TRUE(succeeded(b1));
  ASSERT_TRUE(succeeded(b2));
  EXPECT_NE(b1->get(), b2->get());

  // A builtin platform name and an external profile path are kept in separate
  // key spaces, so one can never be served from the other's entry.
  auto asProfile = getTargetProfile(TargetSelection::externalProfile("Ascend910B1"), stream);
  EXPECT_TRUE(failed(asProfile));
}

TEST(TargetProfileRegistryTest, DoesNotCacheFailures) {
  std::string diagnostic;
  llvm::raw_string_ostream stream(diagnostic);

  EXPECT_TRUE(failed(getTargetProfile(TargetSelection::builtinPlatform("NoSuchPlatform"), stream)));
  EXPECT_FALSE(diagnostic.empty());

  // A failed load must not be cached: retrying should report the failure again
  // and allow a corrected profile to load.
  diagnostic.clear();
  EXPECT_TRUE(failed(getTargetProfile(TargetSelection::builtinPlatform("NoSuchPlatform"), stream)));
  EXPECT_FALSE(diagnostic.empty());
}

TEST(TargetProfileRegistryTest, OptionalSelectionWithoutFlagsYieldsNoProfile) {
  std::string diagnostic;
  llvm::raw_string_ostream stream(diagnostic);

  // No --target/--target-profile set in this test binary.
  auto profile = getSelectedTargetProfile(TargetSelectionRequirement::Optional, stream);
  ASSERT_TRUE(succeeded(profile));
  EXPECT_FALSE(*profile);

  EXPECT_TRUE(failed(getSelectedTargetProfile(TargetSelectionRequirement::Required, stream)));
}

}  // namespace
}  // namespace mlir::ascend
