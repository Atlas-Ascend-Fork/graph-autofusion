/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/TargetSelection.h"

#include <optional>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace mlir::ascend {
namespace {

FailureOr<std::optional<TargetSelection>> Resolve(llvm::StringRef target, llvm::StringRef targetProfile,
                                                  TargetSelectionRequirement requirement, std::string &diagnostic) {
  llvm::raw_string_ostream diagnosticStream(diagnostic);
  auto selection = resolveTargetSelection(target, targetProfile, requirement, diagnosticStream);
  diagnosticStream.flush();
  return selection;
}

TEST(TargetSelectionTest, AllowsMissingSelectionForLegacyPipeline) {
  std::string diagnostic;
  auto selection = Resolve("", "", TargetSelectionRequirement::Optional, diagnostic);

  ASSERT_TRUE(succeeded(selection)) << diagnostic;
  EXPECT_FALSE(selection->has_value());
  EXPECT_TRUE(diagnostic.empty());
}

TEST(TargetSelectionTest, RequiresSelectionForTargetDependentPipeline) {
  std::string diagnostic;
  auto selection = Resolve("", "", TargetSelectionRequirement::Required, diagnostic);

  EXPECT_TRUE(failed(selection));
  EXPECT_NE(diagnostic.find("one of --target or --target-profile is required"), std::string::npos);
}

TEST(TargetSelectionTest, RejectsConflictingSelections) {
  std::string diagnostic;
  auto selection = Resolve("Ascend910B1", "profile.ini", TargetSelectionRequirement::Optional, diagnostic);

  EXPECT_TRUE(failed(selection));
  EXPECT_NE(diagnostic.find("--target and --target-profile cannot be specified together"), std::string::npos);
}

TEST(TargetSelectionTest, ResolvesBuiltinPlatform) {
  std::string diagnostic;
  auto selection = Resolve("Ascend910B1", "", TargetSelectionRequirement::Required, diagnostic);

  ASSERT_TRUE(succeeded(selection)) << diagnostic;
  ASSERT_TRUE(selection->has_value());
  EXPECT_EQ(selection->value().getMode(), TargetSelectionMode::BuiltinPlatform);
  EXPECT_EQ(selection->value().getValue(), "Ascend910B1");
}

TEST(TargetSelectionTest, ResolvesExternalProfile) {
  std::string diagnostic;
  auto selection = Resolve("", "profile.ini", TargetSelectionRequirement::Required, diagnostic);

  ASSERT_TRUE(succeeded(selection)) << diagnostic;
  ASSERT_TRUE(selection->has_value());
  EXPECT_EQ(selection->value().getMode(), TargetSelectionMode::ExternalProfile);
  EXPECT_EQ(selection->value().getValue(), "profile.ini");
}

}  // namespace
}  // namespace mlir::ascend
