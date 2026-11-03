/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/AscendTargetProfile.h"

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace mlir::ascend {
namespace {

TargetProfile CreateProfile(std::string socVersion, NpuArch npuArch = NpuArch::Unknown) {
  TargetIdentity identity{std::move(socVersion), "AscendTest", npuArch};
  CoreTopology topology{32, 16, 16, CubeVectorTopology::Split};
  std::vector<MemoryResource> memories = {
      {MemorySpace::GM, {32LL * 1024 * 1024 * 1024}},
      {MemorySpace::L0C, {128 * 1024}},
      {MemorySpace::UB, {256 * 1024, 32}},
  };
  std::vector<ComputeUnitSpecEntry> computeUnits = {
      {AICoreUnit::Vector, {1800, 256, std::nullopt, std::nullopt, std::nullopt}},
  };
  std::vector<TargetFeatureSpec> features = {
      {TargetFeature::BF16, FeatureState::Available},
      {TargetFeature::InfNan, FeatureState::Unavailable},
  };
  std::vector<ComputeAccess> computeAccesses = {
      {AICoreUnit::Cube, MemorySpace::L0C, AccessMode::Write},
      {AICoreUnit::Vector, MemorySpace::UB, AccessMode::Read},
      {AICoreUnit::Vector, MemorySpace::UB, AccessMode::Write},
  };
  std::vector<TransferEdge> transferEdges = {
      {MemorySpace::L0C, MemorySpace::GM, TransferEngine::Fixpipe, 64},
      {MemorySpace::GM, MemorySpace::UB, TransferEngine::MTE2, 128},
  };
  return TargetProfile({std::move(identity), topology, std::move(memories), std::move(computeUnits),
                        std::move(features), std::move(computeAccesses), std::move(transferEdges)},
                       "profile-fingerprint");
}

TEST(TargetProfileTest, OwnsIdentityAndFingerprintStorage) {
  std::string socVersion = "AscendSynthetic";
  TargetProfile profile = CreateProfile(socVersion);

  socVersion.assign("modified");

  EXPECT_EQ(profile.getIdentity().socVersion, "AscendSynthetic");
  EXPECT_EQ(profile.getIdentity().shortSocVersion, "AscendTest");
  EXPECT_EQ(profile.getFingerprint(), "profile-fingerprint");
}

TEST(TargetProfileTest, ReturnsMemoryBySpace) {
  TargetProfile profile = CreateProfile("AscendSynthetic");

  auto ub = profile.getMemorySpec(MemorySpace::UB);

  ASSERT_TRUE(succeeded(ub));
  ASSERT_TRUE(ub->capacityBytes);
  EXPECT_EQ(*ub->capacityBytes, 256 * 1024);
  EXPECT_EQ(ub->blockBytes, 32);
}

TEST(TargetProfileTest, ReturnsFixedMemoryOwners) {
  EXPECT_EQ(getMemoryOwner(MemorySpace::GM), std::nullopt);
  EXPECT_EQ(getMemoryOwner(MemorySpace::L2), std::nullopt);
  EXPECT_EQ(getMemoryOwner(MemorySpace::L1), AICoreUnit::Cube);
  EXPECT_EQ(getMemoryOwner(MemorySpace::L0A), AICoreUnit::Cube);
  EXPECT_EQ(getMemoryOwner(MemorySpace::L0B), AICoreUnit::Cube);
  EXPECT_EQ(getMemoryOwner(MemorySpace::L0C), AICoreUnit::Cube);
  EXPECT_EQ(getMemoryOwner(MemorySpace::UB), AICoreUnit::Vector);
}

TEST(TargetProfileTest, DistinguishesMissingResourcesFromUnknownSpecs) {
  TargetProfile profile = CreateProfile("AscendSynthetic");

  EXPECT_FALSE(profile.hasMemory(MemorySpace::L2));
  EXPECT_TRUE(failed(profile.getMemorySpec(MemorySpace::L2)));
  EXPECT_TRUE(profile.hasComputeUnit(AICoreUnit::Cube));
  auto cube = profile.getComputeUnitSpec(AICoreUnit::Cube);
  ASSERT_TRUE(succeeded(cube));
  EXPECT_FALSE(cube->frequencyMHz);
  EXPECT_FALSE(cube->vectorCalculationBytes);
}

TEST(TargetProfileTest, PreservesThreeStateFeatureSemantics) {
  TargetProfile profile = CreateProfile("AscendSynthetic");

  EXPECT_EQ(profile.getFeatureState(TargetFeature::BF16), FeatureState::Available);
  EXPECT_TRUE(profile.hasFeature(TargetFeature::BF16));
  EXPECT_EQ(profile.getFeatureState(TargetFeature::InfNan), FeatureState::Unavailable);
  EXPECT_FALSE(profile.hasFeature(TargetFeature::InfNan));
  EXPECT_EQ(profile.getFeatureState(TargetFeature::NDDMA), FeatureState::Unknown);
  EXPECT_FALSE(profile.hasFeature(TargetFeature::NDDMA));
}

TEST(TargetProfileTest, InfersRegBasedApiFamilyFromNpuArch) {
  TargetProfile v2201 = CreateProfile("AscendSynthetic", NpuArch::V2201);
  EXPECT_EQ(v2201.getFeatureState(TargetFeature::RegBasedApiFamily), FeatureState::Unavailable);

  TargetProfile v3510 = CreateProfile("AscendSynthetic", NpuArch::V3510);
  EXPECT_EQ(v3510.getFeatureState(TargetFeature::RegBasedApiFamily), FeatureState::Available);
  EXPECT_TRUE(v3510.hasFeature(TargetFeature::RegBasedApiFamily));

  TargetProfile unknown = CreateProfile("AscendSynthetic", NpuArch::Unknown);
  EXPECT_EQ(unknown.getFeatureState(TargetFeature::RegBasedApiFamily), FeatureState::Unknown);
}

TEST(TargetProfileTest, ExplicitRegBasedApiFamilyStateOverridesArchInference) {
  TargetIdentity identity{"AscendSynthetic", "AscendTest", NpuArch::V3510};
  TargetProfile profile({std::move(identity),
                         CoreTopology{32, 16, 16, CubeVectorTopology::Split},
                         {},
                         {},
                         {{TargetFeature::RegBasedApiFamily, FeatureState::Unavailable}},
                         {},
                         {}},
                        "fingerprint");

  EXPECT_EQ(profile.getFeatureState(TargetFeature::RegBasedApiFamily), FeatureState::Unavailable);
  EXPECT_FALSE(profile.hasFeature(TargetFeature::RegBasedApiFamily));
}

TEST(TargetProfileTest, ReturnsComputeAccessAndDirectTransferFacts) {
  TargetProfile profile = CreateProfile("AscendSynthetic");

  auto vector = profile.getComputeUnitSpec(AICoreUnit::Vector);
  ASSERT_TRUE(succeeded(vector));
  EXPECT_EQ(vector->frequencyMHz, 1800);
  EXPECT_EQ(vector->vectorCalculationBytes, 256);
  EXPECT_TRUE(profile.canAccess(AICoreUnit::Cube, MemorySpace::L0C, AccessMode::Write));
  EXPECT_FALSE(profile.canAccess(AICoreUnit::Cube, MemorySpace::UB, AccessMode::Write));
  EXPECT_TRUE(profile.canAccess(AICoreUnit::Vector, MemorySpace::UB, AccessMode::Read));
  EXPECT_EQ(profile.getComputeAccesses().size(), 3U);
  EXPECT_TRUE(profile.hasDirectTransfer(MemorySpace::L0C, MemorySpace::GM));
  EXPECT_TRUE(profile.hasDirectTransfer(MemorySpace::GM, MemorySpace::UB));
  EXPECT_FALSE(profile.hasDirectTransfer(MemorySpace::L0C, MemorySpace::UB));
  ASSERT_EQ(profile.getTransferEdges().size(), 2U);
  EXPECT_EQ(profile.getTransferEdges().front().bytesPerCycle, 64);
}

TEST(TargetProfileTest, ModelsBaselineAndA5ExecutionCapabilities) {
  const ExecutionSpec simdMemoryBased{ProcessorType::AICore, AICoreUnit::Vector, ExecutionModel::SIMD,
                                      InstructionForm::MemoryBased};
  const ExecutionSpec simdRegBased{ProcessorType::AICore, AICoreUnit::Vector, ExecutionModel::SIMD,
                                   InstructionForm::RegBased};
  const ExecutionSpec simt{ProcessorType::AICore, AICoreUnit::Vector, ExecutionModel::SIMT, std::nullopt};

  TargetProfile baseline = CreateProfile("AscendSynthetic", NpuArch::V2201);
  EXPECT_EQ(baseline.getExecutionSupport(simdMemoryBased), FeatureState::Available);
  EXPECT_EQ(baseline.getExecutionSupport(simdRegBased), FeatureState::Unavailable);
  EXPECT_EQ(baseline.getExecutionSupport(simt), FeatureState::Unavailable);

  TargetProfile a5 = CreateProfile("AscendSynthetic", NpuArch::V3510);
  EXPECT_EQ(a5.getExecutionSupport(simdMemoryBased), FeatureState::Available);
  EXPECT_EQ(a5.getExecutionSupport(simdRegBased), FeatureState::Available);
  EXPECT_EQ(a5.getExecutionSupport(simt), FeatureState::Available);

  TargetProfile unknown = CreateProfile("AscendSynthetic", NpuArch::Unknown);
  EXPECT_EQ(unknown.getExecutionSupport(simdMemoryBased), FeatureState::Available);
  EXPECT_EQ(unknown.getExecutionSupport(simdRegBased), FeatureState::Unknown);
  EXPECT_EQ(unknown.getExecutionSupport(simt), FeatureState::Unknown);
}

TEST(TargetProfileTest, ExecutionSupportIncludesComputeUnitExistence) {
  TargetIdentity identity{"AscendSynthetic", "AscendTest", NpuArch::V3510};
  CoreTopology topology{16, 16, 0, CubeVectorTopology::Split};
  TargetProfile profile({std::move(identity), topology, {}, {}, {}, {}, {}}, "profile-fingerprint");
  const ExecutionSpec simdMemoryBased{ProcessorType::AICore, AICoreUnit::Vector, ExecutionModel::SIMD,
                                      InstructionForm::MemoryBased};
  const ExecutionSpec simt{ProcessorType::AICore, AICoreUnit::Vector, ExecutionModel::SIMT, std::nullopt};

  EXPECT_FALSE(profile.hasComputeUnit(AICoreUnit::Vector));
  EXPECT_TRUE(failed(profile.getComputeUnitSpec(AICoreUnit::Vector)));
  EXPECT_EQ(profile.getExecutionSupport(simdMemoryBased), FeatureState::Unavailable);
  EXPECT_EQ(profile.getExecutionSupport(simt), FeatureState::Unavailable);
}

TEST(TargetProfileTest, RejectsInvalidExecutionSpecCombinations) {
  TargetProfile profile = CreateProfile("AscendSynthetic", NpuArch::V3510);
  const ExecutionSpec simdWithoutForm{ProcessorType::AICore, AICoreUnit::Vector, ExecutionModel::SIMD, std::nullopt};
  const ExecutionSpec simtWithForm{ProcessorType::AICore, AICoreUnit::Vector, ExecutionModel::SIMT,
                                   InstructionForm::RegBased};

  EXPECT_TRUE(failed(validateExecutionSpec(simdWithoutForm)));
  EXPECT_TRUE(failed(validateExecutionSpec(simtWithForm)));
  EXPECT_EQ(profile.getExecutionSupport(simdWithoutForm), FeatureState::Unavailable);
  EXPECT_EQ(profile.getExecutionSupport(simtWithForm), FeatureState::Unavailable);
}

}  // namespace
}  // namespace mlir::ascend
