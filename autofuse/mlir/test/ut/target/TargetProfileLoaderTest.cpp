/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/TargetProfileLoader.h"

#include <algorithm>
#include <array>
#include <string>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace mlir::ascend {
namespace {

constexpr llvm::StringLiteral kSyntheticProfile = R"ini(
[version]
SoC_version=AscendSynthetic
Short_SoC_version=AscendTest

[SoCInfo]
ai_core_cnt=8
cube_core_cnt=4
vector_core_cnt=4
cube_vector_combine=split
memory_size=17179869184
l2_size=67108864

[AICoreSpec]
l1_size=524288
l0_a_size=65536
l0_b_size=65536
l0_c_size=131072
ub_size=262144
ubblock_size=32

[AICoreintrinsicDtypeMap]
999=Intrinsic_synthetic_no_transfer|f16
)ini";

FailureOr<TargetProfile> LoadProfile(llvm::StringRef ini, std::string &diagnostic) {
  llvm::raw_string_ostream diagnosticStream(diagnostic);
  auto profile = TargetProfileLoader::loadFromBuffer(llvm::MemoryBufferRef(ini, "synthetic.ini"), diagnosticStream);
  diagnosticStream.flush();
  return profile;
}

FailureOr<TargetProfile> LoadBuiltinProfile(llvm::StringRef platform, std::string &diagnostic) {
  llvm::raw_string_ostream diagnosticStream(diagnostic);
  auto profile = TargetProfileLoader::loadBuiltin(platform, diagnosticStream);
  diagnosticStream.flush();
  return profile;
}

const TransferEdge *FindTransferEdge(const TargetProfile &profile, MemorySpace source, MemorySpace destination,
                                     TransferEngine engine) {
  for (const TransferEdge &edge : profile.getTransferEdges()) {
    if (edge.source == source && edge.destination == destination && edge.engine == engine) {
      return &edge;
    }
  }
  return nullptr;
}

void ExpectMemoryCapacity(const TargetProfile &profile, MemorySpace space, int64_t capacityBytes,
                          std::optional<int64_t> blockBytes = std::nullopt) {
  auto memory = profile.getMemorySpec(space);
  ASSERT_TRUE(succeeded(memory));
  ASSERT_TRUE(memory->capacityBytes);
  EXPECT_EQ(*memory->capacityBytes, capacityBytes);
  EXPECT_EQ(memory->blockBytes, blockBytes);
}

void ExpectSameHardwareFacts(const TargetProfile &first, const TargetProfile &second) {
  EXPECT_EQ(first.getIdentity().shortSocVersion, second.getIdentity().shortSocVersion);
  EXPECT_EQ(first.getIdentity().npuArch, second.getIdentity().npuArch);
  EXPECT_EQ(first.getCoreTopology().aiCoreCount, second.getCoreTopology().aiCoreCount);
  EXPECT_EQ(first.getCoreTopology().cubeCoreCount, second.getCoreTopology().cubeCoreCount);
  EXPECT_EQ(first.getCoreTopology().vectorCoreCount, second.getCoreTopology().vectorCoreCount);
  EXPECT_EQ(first.getCoreTopology().cubeVectorTopology, second.getCoreTopology().cubeVectorTopology);
  for (MemorySpace space : {MemorySpace::GM, MemorySpace::L2, MemorySpace::L1, MemorySpace::L0A, MemorySpace::L0B,
                            MemorySpace::L0C, MemorySpace::UB}) {
    auto firstMemory = first.getMemorySpec(space);
    auto secondMemory = second.getMemorySpec(space);
    ASSERT_EQ(succeeded(firstMemory), succeeded(secondMemory));
    if (succeeded(firstMemory)) {
      EXPECT_EQ(firstMemory->capacityBytes, secondMemory->capacityBytes);
      EXPECT_EQ(firstMemory->blockBytes, secondMemory->blockBytes);
    }
  }
  ASSERT_EQ(first.getComputeAccesses().size(), second.getComputeAccesses().size());
  for (size_t index = 0; index < first.getComputeAccesses().size(); ++index) {
    EXPECT_EQ(first.getComputeAccesses()[index].unit, second.getComputeAccesses()[index].unit);
    EXPECT_EQ(first.getComputeAccesses()[index].space, second.getComputeAccesses()[index].space);
    EXPECT_EQ(first.getComputeAccesses()[index].mode, second.getComputeAccesses()[index].mode);
  }
  ASSERT_EQ(first.getTransferEdges().size(), second.getTransferEdges().size());
  for (size_t index = 0; index < first.getTransferEdges().size(); ++index) {
    EXPECT_TRUE(first.getTransferEdges()[index].source == second.getTransferEdges()[index].source);
    EXPECT_TRUE(first.getTransferEdges()[index].destination == second.getTransferEdges()[index].destination);
    EXPECT_EQ(first.getTransferEdges()[index].engine, second.getTransferEdges()[index].engine);
    EXPECT_EQ(first.getTransferEdges()[index].bytesPerCycle, second.getTransferEdges()[index].bytesPerCycle);
  }
}

TEST(TargetProfileLoaderTest, LoadsRequiredIdentityCoreAndMemoryFacts) {
  std::string diagnostic;
  auto profile = LoadProfile(kSyntheticProfile, diagnostic);

  ASSERT_TRUE(succeeded(profile)) << diagnostic;
  EXPECT_EQ(profile->getIdentity().socVersion, "AscendSynthetic");
  EXPECT_EQ(profile->getIdentity().shortSocVersion, "AscendTest");
  EXPECT_EQ(profile->getCoreTopology().aiCoreCount, 8);
  EXPECT_EQ(profile->getCoreTopology().cubeCoreCount, 4);
  EXPECT_EQ(profile->getCoreTopology().vectorCoreCount, 4);
  EXPECT_EQ(profile->getCoreTopology().cubeVectorTopology, CubeVectorTopology::Split);

  auto globalMemory = profile->getMemorySpec(MemorySpace::GM);
  auto l2 = profile->getMemorySpec(MemorySpace::L2);
  auto l1 = profile->getMemorySpec(MemorySpace::L1);
  auto l0a = profile->getMemorySpec(MemorySpace::L0A);
  auto l0b = profile->getMemorySpec(MemorySpace::L0B);
  auto l0c = profile->getMemorySpec(MemorySpace::L0C);
  auto unifiedBuffer = profile->getMemorySpec(MemorySpace::UB);
  ASSERT_TRUE(succeeded(globalMemory));
  ASSERT_TRUE(succeeded(l2));
  ASSERT_TRUE(succeeded(l1));
  ASSERT_TRUE(succeeded(l0a));
  ASSERT_TRUE(succeeded(l0b));
  ASSERT_TRUE(succeeded(l0c));
  ASSERT_TRUE(succeeded(unifiedBuffer));
  EXPECT_EQ(globalMemory->capacityBytes, std::optional<int64_t>(17179869184LL));
  EXPECT_EQ(l2->capacityBytes, std::optional<int64_t>(67108864));
  EXPECT_EQ(l1->capacityBytes, std::optional<int64_t>(524288));
  EXPECT_EQ(l0a->capacityBytes, std::optional<int64_t>(65536));
  EXPECT_EQ(l0b->capacityBytes, std::optional<int64_t>(65536));
  EXPECT_EQ(l0c->capacityBytes, std::optional<int64_t>(131072));
  EXPECT_EQ(unifiedBuffer->capacityBytes, std::optional<int64_t>(262144));
  EXPECT_EQ(unifiedBuffer->blockBytes, 32);
  EXPECT_TRUE(diagnostic.empty());
}

TEST(TargetProfileLoaderTest, LeavesEmptyGlobalMemoryCapacityUnmodeled) {
  std::string emptyGlobalMemory(kSyntheticProfile);
  size_t valuePosition = emptyGlobalMemory.find("memory_size=17179869184");
  ASSERT_NE(valuePosition, std::string::npos);
  emptyGlobalMemory.replace(valuePosition, std::string("memory_size=17179869184").size(), "memory_size=");

  std::string diagnostic;
  auto profile = LoadProfile(emptyGlobalMemory, diagnostic);

  ASSERT_TRUE(succeeded(profile)) << diagnostic;
  EXPECT_TRUE(profile->hasMemory(MemorySpace::GM));
  auto globalMemory = profile->getMemorySpec(MemorySpace::GM);
  ASSERT_TRUE(succeeded(globalMemory));
  EXPECT_FALSE(globalMemory->capacityBytes);
  EXPECT_TRUE(profile->hasDirectTransfer(MemorySpace::GM, MemorySpace::UB));
  ExpectMemoryCapacity(*profile, MemorySpace::L2, 67108864);
}

TEST(TargetProfileLoaderTest, KeepsLocalMemoryResourcesWhenSpecsAreMissing) {
  std::string missingSpecs(kSyntheticProfile);
  for (llvm::StringRef field : {"l0_c_size=131072\n", "ub_size=262144\n", "ubblock_size=32\n"}) {
    size_t position = missingSpecs.find(field.str());
    ASSERT_NE(position, std::string::npos);
    missingSpecs.erase(position, field.size());
  }

  std::string diagnostic;
  auto profile = LoadProfile(missingSpecs, diagnostic);

  ASSERT_TRUE(succeeded(profile)) << diagnostic;
  EXPECT_TRUE(profile->hasMemory(MemorySpace::L0C));
  EXPECT_TRUE(profile->hasMemory(MemorySpace::UB));
  auto l0c = profile->getMemorySpec(MemorySpace::L0C);
  auto ub = profile->getMemorySpec(MemorySpace::UB);
  ASSERT_TRUE(succeeded(l0c));
  ASSERT_TRUE(succeeded(ub));
  EXPECT_FALSE(l0c->capacityBytes);
  EXPECT_FALSE(ub->capacityBytes);
  EXPECT_FALSE(ub->blockBytes);
  EXPECT_TRUE(profile->canAccess(AICoreUnit::Cube, MemorySpace::L0C, AccessMode::Write));
  EXPECT_TRUE(profile->canAccess(AICoreUnit::Vector, MemorySpace::UB, AccessMode::Read));
  EXPECT_TRUE(profile->hasDirectTransfer(MemorySpace::GM, MemorySpace::UB));
}

TEST(TargetProfileLoaderTest, RequiresCoreCountFields) {
  for (llvm::StringRef field : {"cube_core_cnt=4\n", "vector_core_cnt=4\n"}) {
    SCOPED_TRACE(field.str());
    std::string missingCount(kSyntheticProfile);
    size_t position = missingCount.find(field.str());
    ASSERT_NE(position, std::string::npos);
    missingCount.erase(position, field.size());

    std::string diagnostic;
    EXPECT_TRUE(failed(LoadProfile(missingCount, diagnostic)));
    EXPECT_NE(diagnostic.find(("missing required field SoCInfo." + field.split('=').first).str()), std::string::npos);
  }
}

TEST(TargetProfileLoaderTest, ZeroCoreCountRemovesOwnedResourcesAndCapabilities) {
  std::string noCube(kSyntheticProfile);
  size_t position = noCube.find("cube_core_cnt=4");
  ASSERT_NE(position, std::string::npos);
  noCube.replace(position, std::string("cube_core_cnt=4").size(), "cube_core_cnt=0");

  std::string diagnostic;
  auto cubeProfile = LoadProfile(noCube, diagnostic);
  ASSERT_TRUE(succeeded(cubeProfile)) << diagnostic;
  EXPECT_FALSE(cubeProfile->hasComputeUnit(AICoreUnit::Cube));
  EXPECT_TRUE(failed(cubeProfile->getComputeUnitSpec(AICoreUnit::Cube)));
  EXPECT_FALSE(cubeProfile->hasMemory(MemorySpace::L0C));
  EXPECT_FALSE(cubeProfile->canAccess(AICoreUnit::Cube, MemorySpace::L0C, AccessMode::Write));

  std::string noVector(kSyntheticProfile);
  position = noVector.find("vector_core_cnt=4");
  ASSERT_NE(position, std::string::npos);
  noVector.replace(position, std::string("vector_core_cnt=4").size(), "vector_core_cnt=0");
  noVector.append("\n[AICoreintrinsicDtypeMap]\n0=Intrinsic_data_move_out2ub|f16\n");

  diagnostic.clear();
  auto vectorProfile = LoadProfile(noVector, diagnostic);
  ASSERT_TRUE(succeeded(vectorProfile)) << diagnostic;
  EXPECT_FALSE(vectorProfile->hasComputeUnit(AICoreUnit::Vector));
  EXPECT_TRUE(failed(vectorProfile->getComputeUnitSpec(AICoreUnit::Vector)));
  EXPECT_FALSE(vectorProfile->hasMemory(MemorySpace::UB));
  EXPECT_FALSE(vectorProfile->canAccess(AICoreUnit::Vector, MemorySpace::UB, AccessMode::Read));
  EXPECT_FALSE(vectorProfile->hasDirectTransfer(MemorySpace::GM, MemorySpace::UB));
  EXPECT_EQ(FindTransferEdge(*vectorProfile, MemorySpace::GM, MemorySpace::UB, TransferEngine::MTE2), nullptr);
}

TEST(TargetProfileLoaderTest, ReturnsUnknownComputeSpecsForExistingUnits) {
  std::string diagnostic;
  auto profile = LoadProfile(kSyntheticProfile, diagnostic);

  ASSERT_TRUE(succeeded(profile)) << diagnostic;
  EXPECT_TRUE(profile->hasComputeUnit(AICoreUnit::Cube));
  EXPECT_TRUE(profile->hasComputeUnit(AICoreUnit::Vector));
  auto cube = profile->getComputeUnitSpec(AICoreUnit::Cube);
  auto vector = profile->getComputeUnitSpec(AICoreUnit::Vector);
  ASSERT_TRUE(succeeded(cube));
  ASSERT_TRUE(succeeded(vector));
  EXPECT_FALSE(cube->frequencyMHz);
  EXPECT_FALSE(vector->frequencyMHz);
  EXPECT_FALSE(vector->vectorCalculationBytes);
}

TEST(TargetProfileLoaderTest, RejectsMissingIntrinsicMapSection) {
  std::string missingMap(kSyntheticProfile);
  size_t section = missingMap.find("[AICoreintrinsicDtypeMap]");
  ASSERT_NE(section, std::string::npos);
  missingMap.erase(section);

  std::string diagnostic;
  EXPECT_TRUE(failed(LoadProfile(missingMap, diagnostic)));
  EXPECT_NE(diagnostic.find("missing required section AICoreintrinsicDtypeMap"), std::string::npos);
}

TEST(TargetProfileLoaderTest, RejectsMissingRequiredField) {
  std::string diagnostic;
  auto profile = LoadProfile("[version]\nSoC_version=AscendSynthetic\n", diagnostic);

  EXPECT_TRUE(failed(profile));
  EXPECT_NE(diagnostic.find("SoCInfo.ai_core_cnt"), std::string::npos);
}

TEST(TargetProfileLoaderTest, RejectsInvalidSocVersion) {
  const std::array<std::string, 3> invalidSocVersions = {
      "Ascend910B1/../../unexpected",
      "Ascend 910B1",
      std::string(129, 'A'),
  };

  for (const std::string &socVersion : invalidSocVersions) {
    SCOPED_TRACE(socVersion);
    std::string invalidProfile(kSyntheticProfile);
    size_t valuePosition = invalidProfile.find("SoC_version=AscendSynthetic");
    ASSERT_NE(valuePosition, std::string::npos);
    invalidProfile.replace(valuePosition, std::string("SoC_version=AscendSynthetic").size(),
                           "SoC_version=" + socVersion);

    std::string diagnostic;
    EXPECT_TRUE(failed(LoadProfile(invalidProfile, diagnostic)));
    EXPECT_NE(diagnostic.find("invalid platform name in field version.SoC_version"), std::string::npos);
  }
}

TEST(TargetProfileLoaderTest, AcceptsMaximumLengthSocVersion) {
  std::string maximumLengthSocVersion(128, 'A');
  std::string profileText(kSyntheticProfile);
  size_t valuePosition = profileText.find("SoC_version=AscendSynthetic");
  ASSERT_NE(valuePosition, std::string::npos);
  profileText.replace(valuePosition, std::string("SoC_version=AscendSynthetic").size(),
                      "SoC_version=" + maximumLengthSocVersion);

  std::string diagnostic;
  auto profile = LoadProfile(profileText, diagnostic);

  ASSERT_TRUE(succeeded(profile)) << diagnostic;
  EXPECT_EQ(profile->getIdentity().socVersion, maximumLengthSocVersion);
}

TEST(TargetProfileLoaderTest, RejectsInvalidAndNonPositiveInteger) {
  std::string invalidInteger(kSyntheticProfile);
  size_t valuePosition = invalidInteger.find("ai_core_cnt=8");
  ASSERT_NE(valuePosition, std::string::npos);
  invalidInteger.replace(valuePosition, std::string("ai_core_cnt=8").size(), "ai_core_cnt=invalid");

  std::string diagnostic;
  EXPECT_TRUE(failed(LoadProfile(invalidInteger, diagnostic)));
  EXPECT_NE(diagnostic.find("invalid integer"), std::string::npos);

  std::string zeroCapacity(kSyntheticProfile);
  valuePosition = zeroCapacity.find("ub_size=262144");
  ASSERT_NE(valuePosition, std::string::npos);
  zeroCapacity.replace(valuePosition, std::string("ub_size=262144").size(), "ub_size=0");
  diagnostic.clear();
  EXPECT_TRUE(failed(LoadProfile(zeroCapacity, diagnostic)));
  EXPECT_NE(diagnostic.find("must be greater than zero"), std::string::npos);
}

TEST(TargetProfileLoaderTest, RejectsDuplicateSectionKey) {
  std::string duplicate(kSyntheticProfile);
  duplicate.append("\n[AICoreSpec]\nub_size=524288\n");

  std::string diagnostic;
  EXPECT_TRUE(failed(LoadProfile(duplicate, diagnostic)));
  EXPECT_NE(diagnostic.find("duplicate field AICoreSpec.ub_size"), std::string::npos);
}

TEST(TargetProfileLoaderTest, RejectsOversizedBuffer) {
  std::string oversizedProfile(256 * 1024 + 1, 'x');

  std::string diagnostic;
  EXPECT_TRUE(failed(LoadProfile(oversizedProfile, diagnostic)));
  EXPECT_NE(diagnostic.find("target profile exceeds maximum size of 262144 bytes"), std::string::npos);
}

TEST(TargetProfileLoaderTest, RejectsOversizedExternalFile) {
  llvm::SmallString<128> path;
  int fileDescriptor = -1;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("oversized-target-profile", "ini", fileDescriptor, path));
  {
    llvm::raw_fd_ostream stream(fileDescriptor, true);
    stream << std::string(256 * 1024 + 1, 'x');
  }

  std::string diagnostic;
  llvm::raw_string_ostream diagnosticStream(diagnostic);
  auto profile = TargetProfileLoader::load(TargetSelection::externalProfile(path.str().str()), diagnosticStream);
  diagnosticStream.flush();
  EXPECT_TRUE(failed(profile));
  EXPECT_NE(diagnostic.find("target profile exceeds maximum size of 262144 bytes"), std::string::npos);

  EXPECT_FALSE(llvm::sys::fs::remove(path));
}

TEST(TargetProfileLoaderTest, RejectsOverlongLine) {
  std::string overlongLine = "[version]\n" + std::string(4097, 'x') + "\n";

  std::string diagnostic;
  EXPECT_TRUE(failed(LoadProfile(overlongLine, diagnostic)));
  EXPECT_NE(diagnostic.find("line 2: exceeds maximum length of 4096 bytes"), std::string::npos);
}

TEST(TargetProfileLoaderTest, RejectsTooManyFields) {
  std::string tooManyFields = "[section]\n";
  for (size_t index = 0; index <= 4096; ++index) {
    tooManyFields.append("field").append(std::to_string(index)).append("=value\n");
  }

  std::string diagnostic;
  EXPECT_TRUE(failed(LoadProfile(tooManyFields, diagnostic)));
  EXPECT_NE(diagnostic.find("target profile exceeds maximum field count of 4096"), std::string::npos);
}

TEST(TargetProfileLoaderTest, EscapesControlCharactersInDiagnosticIdentifier) {
  std::string diagnostic;
  llvm::raw_string_ostream diagnosticStream(diagnostic);
  auto profile = TargetProfileLoader::load(TargetSelection::externalProfile("missing.ini\nINJECTED: fake error"),
                                           diagnosticStream);
  diagnosticStream.flush();

  EXPECT_TRUE(failed(profile));
  EXPECT_NE(diagnostic.find("missing.ini\\0AINJECTED: fake error: target profile is not a readable regular file"),
            std::string::npos);
  EXPECT_EQ(std::count(diagnostic.begin(), diagnostic.end(), '\n'), 1);
}

TEST(TargetProfileLoaderTest, FingerprintIsIndependentOfFieldOrder) {
  constexpr llvm::StringLiteral reordered = R"ini(
[AICoreSpec]
ubblock_size=32
ub_size=262144
l0_c_size=131072
l0_b_size=65536
l0_a_size=65536
l1_size=524288
[SoCInfo]
l2_size=67108864
memory_size=17179869184
cube_vector_combine=split
vector_core_cnt=4
cube_core_cnt=4
ai_core_cnt=8
[version]
Short_SoC_version=AscendTest
SoC_version=AscendSynthetic
[AICoreintrinsicDtypeMap]
999=Intrinsic_synthetic_no_transfer|f16
)ini";

  std::string firstDiagnostic;
  std::string secondDiagnostic;
  auto first = LoadProfile(kSyntheticProfile, firstDiagnostic);
  auto second = LoadProfile(reordered, secondDiagnostic);

  ASSERT_TRUE(succeeded(first)) << firstDiagnostic;
  ASSERT_TRUE(succeeded(second)) << secondDiagnostic;
  EXPECT_EQ(first->getFingerprint(), second->getFingerprint());
  EXPECT_EQ(first->getFingerprint().size(), 64U);
}

TEST(TargetProfileLoaderTest, FingerprintChangesWithPublishedFact) {
  std::string changed(kSyntheticProfile);
  size_t valuePosition = changed.find("ai_core_cnt=8");
  ASSERT_NE(valuePosition, std::string::npos);
  changed.replace(valuePosition, std::string("ai_core_cnt=8").size(), "ai_core_cnt=16");

  std::string firstDiagnostic;
  std::string secondDiagnostic;
  auto first = LoadProfile(kSyntheticProfile, firstDiagnostic);
  auto second = LoadProfile(changed, secondDiagnostic);

  ASSERT_TRUE(succeeded(first)) << firstDiagnostic;
  ASSERT_TRUE(succeeded(second)) << secondDiagnostic;
  EXPECT_NE(first->getFingerprint(), second->getFingerprint());
}

TEST(TargetProfileLoaderTest, LoadsCanonicalFactsForAliasProfiles) {
  std::string diagnostic;
  auto profile = LoadBuiltinProfile("Ascend910B2C", diagnostic);

  ASSERT_TRUE(succeeded(profile)) << diagnostic;
  EXPECT_EQ(profile->getIdentity().socVersion, "Ascend910B2C");
  ExpectMemoryCapacity(*profile, MemorySpace::GM, 68719476736);

  diagnostic.clear();
  auto alias = LoadBuiltinProfile("Ascend910B4-1", diagnostic);
  ASSERT_TRUE(succeeded(alias)) << diagnostic;
  EXPECT_EQ(alias->getIdentity().socVersion, "Ascend910B4-1");
  ExpectMemoryCapacity(*alias, MemorySpace::L2, 100663296);
}

TEST(TargetProfileLoaderTest, EnablesRegBasedApiFamilyForAscend950Dt) {
  std::string diagnostic;
  auto profile = LoadBuiltinProfile("Ascend950DT_950x", diagnostic);

  ASSERT_TRUE(succeeded(profile)) << diagnostic;
  EXPECT_EQ(profile->getIdentity().npuArch, NpuArch::V3510);
  EXPECT_TRUE(profile->hasFeature(TargetFeature::RegBasedApiFamily));
}

TEST(TargetProfileLoaderTest, LoadsRepresentativeFamilyFacts) {
  struct ExpectedProfile {
    llvm::StringLiteral platform;
    llvm::StringLiteral shortSocVersion;
    NpuArch npuArch;
    CubeVectorTopology topology;
    bool hasDirectL0CToUB;
    int64_t aiCoreCount;
    int64_t cubeCoreCount;
    int64_t vectorCoreCount;
    std::optional<int64_t> gmBytes;
    int64_t l2Bytes;
    int64_t l1Bytes;
    int64_t l0cBytes;
    int64_t ubBytes;
  };
  const std::array expectedProfiles = {
      ExpectedProfile{"Ascend910B1", "Ascend910B", NpuArch::V2201, CubeVectorTopology::Split, false, 24, 24, 48,
                      68719476736, 201326592, 524288, 131072, 196608},
      ExpectedProfile{"Ascend910_9381", "Ascend910_93", NpuArch::V2201, CubeVectorTopology::Split, false, 24, 24, 48,
                      68719476736, 201326592, 524288, 131072, 196608},
      ExpectedProfile{"Ascend950DT_9571", "Ascend950", NpuArch::V3510, CubeVectorTopology::Split, true, 28, 28, 56,
                      154618822656, 134217728, 524288, 262144, 253952},
  };

  for (const ExpectedProfile &expected : expectedProfiles) {
    SCOPED_TRACE(expected.platform.str());
    std::string diagnostic;
    auto profile = LoadBuiltinProfile(expected.platform, diagnostic);
    ASSERT_TRUE(succeeded(profile)) << diagnostic;
    EXPECT_EQ(profile->getIdentity().shortSocVersion, expected.shortSocVersion);
    EXPECT_EQ(profile->getIdentity().npuArch, expected.npuArch);
    EXPECT_EQ(profile->getCoreTopology().aiCoreCount, expected.aiCoreCount);
    EXPECT_EQ(profile->getCoreTopology().cubeCoreCount, expected.cubeCoreCount);
    EXPECT_EQ(profile->getCoreTopology().vectorCoreCount, expected.vectorCoreCount);
    EXPECT_EQ(profile->getCoreTopology().cubeVectorTopology, expected.topology);
    const MemorySpace l0c = MemorySpace::L0C;
    const MemorySpace ub = MemorySpace::UB;
    const MemorySpace gm = MemorySpace::GM;
    EXPECT_TRUE(profile->canAccess(AICoreUnit::Cube, l0c, AccessMode::Write));
    EXPECT_TRUE(profile->canAccess(AICoreUnit::Vector, ub, AccessMode::Read));
    EXPECT_TRUE(profile->canAccess(AICoreUnit::Vector, ub, AccessMode::Write));
    EXPECT_TRUE(profile->hasDirectTransfer(l0c, gm));
    EXPECT_TRUE(profile->hasDirectTransfer(gm, ub));
    EXPECT_EQ(profile->hasDirectTransfer(l0c, ub), expected.hasDirectL0CToUB);
    if (expected.gmBytes) {
      ExpectMemoryCapacity(*profile, MemorySpace::GM, *expected.gmBytes);
    } else {
      EXPECT_TRUE(failed(profile->getMemorySpec(MemorySpace::GM)));
    }
    ExpectMemoryCapacity(*profile, MemorySpace::L2, expected.l2Bytes);
    ExpectMemoryCapacity(*profile, MemorySpace::L1, expected.l1Bytes);
    ExpectMemoryCapacity(*profile, MemorySpace::L0A, 65536);
    ExpectMemoryCapacity(*profile, MemorySpace::L0B, 65536);
    ExpectMemoryCapacity(*profile, MemorySpace::L0C, expected.l0cBytes);
    ExpectMemoryCapacity(*profile, MemorySpace::UB, expected.ubBytes, 32);
  }
}

TEST(TargetProfileLoaderTest, MapsMissingAndUnknownNpuArchToUnknown) {
  std::string diagnostic;
  auto missing = LoadProfile(kSyntheticProfile, diagnostic);
  ASSERT_TRUE(succeeded(missing)) << diagnostic;
  EXPECT_EQ(missing->getIdentity().npuArch, NpuArch::Unknown);

  std::string unknownProfile(kSyntheticProfile);
  unknownProfile.insert(unknownProfile.find("Short_SoC_version"), "NpuArch=9999\n");
  diagnostic.clear();
  auto unknown = LoadProfile(unknownProfile, diagnostic);
  ASSERT_TRUE(succeeded(unknown)) << diagnostic;
  EXPECT_EQ(unknown->getIdentity().npuArch, NpuArch::Unknown);
}

TEST(TargetProfileLoaderTest, BuildsDirectTransfersFromIntrinsicMap) {
  std::string availableProfile(kSyntheticProfile);
  availableProfile.append(R"ini(
[AICoreintrinsicDtypeMap]
0=Intrinsic_fix_pipe_l0c2out|f32,s32,f16
1=Intrinsic_fix_pipe_l0c2ub|f32,s32,f16
)ini");
  std::string diagnostic;
  auto available = LoadProfile(availableProfile, diagnostic);
  ASSERT_TRUE(succeeded(available)) << diagnostic;
  const MemorySpace gm = MemorySpace::GM;
  const MemorySpace l0c = MemorySpace::L0C;
  const MemorySpace ub = MemorySpace::UB;
  EXPECT_TRUE(available->hasDirectTransfer(l0c, gm));
  EXPECT_TRUE(available->hasDirectTransfer(gm, ub));
  EXPECT_TRUE(available->hasDirectTransfer(l0c, ub));

  std::string unavailableProfile(kSyntheticProfile);
  unavailableProfile.append("\n[AICoreintrinsicDtypeMap]\n0=Intrinsic_fix_pipe_l0c2out|f32,s32,f16\n");
  diagnostic.clear();
  auto unavailable = LoadProfile(unavailableProfile, diagnostic);
  ASSERT_TRUE(succeeded(unavailable)) << diagnostic;
  EXPECT_TRUE(unavailable->hasDirectTransfer(l0c, gm));
  EXPECT_TRUE(unavailable->hasDirectTransfer(gm, ub));
  EXPECT_FALSE(unavailable->hasDirectTransfer(l0c, ub));

  diagnostic.clear();
  auto unknown = LoadProfile(kSyntheticProfile, diagnostic);
  ASSERT_TRUE(succeeded(unknown)) << diagnostic;
  EXPECT_TRUE(unknown->hasDirectTransfer(gm, ub));
  EXPECT_TRUE(unknown->hasDirectTransfer(ub, gm));
  EXPECT_FALSE(unknown->hasDirectTransfer(l0c, gm));
  EXPECT_FALSE(unknown->hasDirectTransfer(l0c, ub));
}

TEST(TargetProfileLoaderTest, MapsDataMoveIntrinsicsToConcreteMtePipelines) {
  std::string profileText(kSyntheticProfile);
  profileText.append(R"ini(
[AICoreintrinsicDtypeMap]
0=Intrinsic_data_move_l12l0a|f16
1=Intrinsic_data_move_out2ub|f16
2=Intrinsic_data_move_ub2out|f16
3=Intrinsic_fix_pipe_l0c2ub|f16
)ini");
  std::string diagnostic;
  auto profile = LoadProfile(profileText, diagnostic);
  ASSERT_TRUE(succeeded(profile)) << diagnostic;

  EXPECT_NE(FindTransferEdge(*profile, MemorySpace::L1, MemorySpace::L0A, TransferEngine::MTE1), nullptr);
  EXPECT_NE(FindTransferEdge(*profile, MemorySpace::GM, MemorySpace::UB, TransferEngine::MTE2), nullptr);
  EXPECT_NE(FindTransferEdge(*profile, MemorySpace::UB, MemorySpace::GM, TransferEngine::MTE3), nullptr);
  EXPECT_NE(FindTransferEdge(*profile, MemorySpace::L0C, MemorySpace::UB, TransferEngine::Fixpipe), nullptr);
}

TEST(TargetProfileLoaderTest, TransferRateDoesNotCreateMissingDirectEdge) {
  std::string profileText(kSyntheticProfile);
  profileText.append(R"ini(
[AICoreintrinsicDtypeMap]
0=Intrinsic_fix_pipe_l0c2out|f32,s32,f16

[AICoreMemoryRates]
l0_c_to_ub_rate=256
)ini");
  std::string diagnostic;
  auto profile = LoadProfile(profileText, diagnostic);
  ASSERT_TRUE(succeeded(profile)) << diagnostic;

  const MemorySpace l0c = MemorySpace::L0C;
  const MemorySpace ub = MemorySpace::UB;
  EXPECT_FALSE(profile->hasDirectTransfer(l0c, ub));
  EXPECT_EQ(FindTransferEdge(*profile, l0c, ub, TransferEngine::Fixpipe), nullptr);
}

TEST(TargetProfileLoaderTest, TransferRateAnnotatesDeclaredDirectEdge) {
  std::string profileText(kSyntheticProfile);
  profileText.append(R"ini(
[AICoreintrinsicDtypeMap]
0=Intrinsic_fix_pipe_l0c2ub|f32,s32,f16

[AICoreMemoryRates]
l0_c_to_ub_rate=256
)ini");
  std::string diagnostic;
  auto profile = LoadProfile(profileText, diagnostic);
  ASSERT_TRUE(succeeded(profile)) << diagnostic;

  const MemorySpace l0c = MemorySpace::L0C;
  const MemorySpace ub = MemorySpace::UB;
  const TransferEdge *edge = FindTransferEdge(*profile, l0c, ub, TransferEngine::Fixpipe);
  ASSERT_NE(edge, nullptr);
  EXPECT_EQ(edge->bytesPerCycle, 256);
}

TEST(TargetProfileLoaderTest, DirectTransferChangesFingerprint) {
  std::string availableProfile(kSyntheticProfile);
  availableProfile.append("\n[AICoreintrinsicDtypeMap]\n0=Intrinsic_fix_pipe_l0c2ub|f32,s32,f16\n");
  std::string unavailableProfile(kSyntheticProfile);
  unavailableProfile.append("\n[AICoreintrinsicDtypeMap]\n0=Intrinsic_fix_pipe_l0c2out|f32,s32,f16\n");

  std::string availableDiagnostic;
  std::string unavailableDiagnostic;
  auto available = LoadProfile(availableProfile, availableDiagnostic);
  auto unavailable = LoadProfile(unavailableProfile, unavailableDiagnostic);

  ASSERT_TRUE(succeeded(available)) << availableDiagnostic;
  ASSERT_TRUE(succeeded(unavailable)) << unavailableDiagnostic;
  EXPECT_NE(available->getFingerprint(), unavailable->getFingerprint());
}

TEST(TargetProfileLoaderTest, ReusesHardwareFactsAcrossAliasesWithoutReusingIdentity) {
  std::string canonicalDiagnostic;
  std::string aliasDiagnostic;
  auto canonical = LoadBuiltinProfile("Ascend950DT_9582", canonicalDiagnostic);
  auto alias = LoadBuiltinProfile("Ascend950DT_9582x", aliasDiagnostic);
  ASSERT_TRUE(succeeded(canonical)) << canonicalDiagnostic;
  ASSERT_TRUE(succeeded(alias)) << aliasDiagnostic;
  EXPECT_NE(canonical->getIdentity().socVersion, alias->getIdentity().socVersion);
  EXPECT_NE(canonical->getFingerprint(), alias->getFingerprint());
  ExpectSameHardwareFacts(*canonical, *alias);
}

TEST(TargetProfileLoaderTest, RejectsUnknownAndInvalidBuiltinPlatform) {
  std::string diagnostic;
  EXPECT_TRUE(failed(LoadBuiltinProfile("Ascend910A", diagnostic)));
  EXPECT_NE(diagnostic.find("unknown builtin target platform"), std::string::npos);

  diagnostic.clear();
  EXPECT_TRUE(failed(LoadBuiltinProfile("../Ascend910B1", diagnostic)));
  EXPECT_NE(diagnostic.find("invalid builtin target platform name"), std::string::npos);
}

}  // namespace
}  // namespace mlir::ascend
