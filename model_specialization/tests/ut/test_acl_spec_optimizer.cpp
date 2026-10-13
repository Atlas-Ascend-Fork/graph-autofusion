/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_acl_spec_optimizer.cpp
 * \brief Host unit tests for model specialization and task rollback.
 */

#include <sstream>

// Opens the optimizer internals to the tests, replacing a production friend declaration that
// existed only for them. Must precede the header it applies to.
#define private public
#define protected public
#include "acl_spec_optimizer.h"
#undef private
#undef protected
#include "feature_manager.h"
#include "spec_scope_launch.h"
#include "runtime/rt_external_kernel.h"
#include "aclmodel_specialization.h"
#include "ut_common_stubs.h"
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>
#include <mockcpp/ApiHook.h>
#include <new>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace model_spec {

class AclSpecOptimizerTestAccess {
 public:
  aclError Optimize(aclmdlRI model, const FeatureManager &manager) {
    AclSpecOptimizer optimizer;
    const aclError result = optimizer.Init(model);
    if (result != ACL_SUCCESS) {
      return result;
    }
    return optimizer.Optimize(model, manager);
  }

  aclError Collect(aclmdlRI model, const FeatureManager &manager, std::string_view directory,
                   std::vector<std::unique_ptr<KernelSpecRequest>> &requests, std::vector<aclmdlRITask> &scopeMarkers) {
    return AclSpecOptimizer().Collect(model, manager, directory, requests, scopeMarkers);
  }
  aclError UpdateTasks(aclmdlRI model, const std::vector<std::unique_ptr<KernelSpecRequest>> &requests) {
    return AclSpecOptimizer().UpdateTasks(model, requests);
  }
};

namespace {
struct TaskState {
  aclmdlRITaskParams params{};
  aclmdlRITaskType type = ACL_MODEL_RI_TASK_KERNEL;
  aclError typeError = ACL_SUCCESS;
  aclError paramsError = ACL_SUCCESS;
};

std::vector<aclError> setResults;
std::vector<aclError> updateResults;
std::vector<aclmdlRITask> touched;
size_t updateCount = 0;

aclError SetTaskParams(aclmdlRITask task, aclmdlRITaskParams *params) {
  const size_t index = touched.size();
  touched.push_back(task);
  static_cast<TaskState *>(task)->params = *params;
  return index < setResults.size() ? setResults[index] : ACL_SUCCESS;
}

aclError UpdateModel(aclmdlRI) {
  const size_t index = updateCount++;
  return index < updateResults.size() ? updateResults[index] : ACL_SUCCESS;
}

size_t identityParamsQueries = 0;
size_t identityTypeQueries = 0;

aclError GetTaskParams(aclmdlRITask task, aclmdlRITaskParams *params) {
  ++identityParamsQueries;
  *params = static_cast<TaskState *>(task)->params;
  return static_cast<TaskState *>(task)->paramsError;
}

aclError GetTaskType(aclmdlRITask task, aclmdlRITaskType *type) {
  ++identityTypeQueries;
  const auto &state = *static_cast<TaskState *>(task);
  *type = state.type;
  return state.typeError;
}

class AclSpecOptimizerUpdateTest : public testing::Test {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
    MOCKER(aclmdlRITaskSetParams).stubs().will(invoke(SetTaskParams));
    MOCKER(aclmdlRIUpdate).stubs().will(invoke(UpdateModel));
    setResults.clear();
    updateResults.clear();
    touched.clear();
    updateCount = 0;
    for (size_t index = 0; index < tasks.size(); ++index) {
      auto &params = tasks[index].params;
      params = {};
      params.type = ACL_MODEL_RI_TASK_KERNEL;
      params.taskGrp = reinterpret_cast<aclrtTaskGrp>(0x100 + index);
      params.opInfoPtr = &tasks[index];
      params.opInfoSize = 17;
      params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x200 + index);
      params.kernelTaskParams.args = &tasks[index];
      params.kernelTaskParams.argsSize = sizeof(TaskState);
      params.kernelTaskParams.numBlocks = 19;
      params.kernelTaskParams.isHostArgs = 1;
      params.kernelTaskParams.cfg = &config;
      auto request = std::make_unique<KernelSpecRequest>();
      request->task = &tasks[index];
      request->originalParams = params;
      request->specFuncHandle = reinterpret_cast<aclrtFuncHandle>(0x300 + index);
      rewrites.push_back(std::move(request));
    }
  }

  void TearDown() override {
    GlobalMockObject::verify();
    SkUtResetTestControls();
  }

  aclmdlRI model = reinterpret_cast<aclmdlRI>(0x123);
  aclrtLaunchKernelCfg config{};
  std::array<TaskState, 3> tasks{};
  std::vector<std::unique_ptr<KernelSpecRequest>> rewrites;
  AclSpecOptimizerTestAccess updater;
};
}  // namespace

TEST_F(AclSpecOptimizerUpdateTest, EmptyListCommitsWithoutTouchingAnyTask) {
  const auto result = updater.UpdateTasks(model, {});
  EXPECT_EQ(result, ACL_SUCCESS);
  EXPECT_TRUE(touched.empty());
  // An empty list no longer turns the commit away: staging nothing and committing it leaves the
  // model as it was, so the screening the caller used to do here earns nothing.
  EXPECT_EQ(updateCount, 1U);
}

TEST_F(AclSpecOptimizerUpdateTest, SuccessPreservesAllOtherFieldsAndUpdatesOnce) {
  const auto result = updater.UpdateTasks(model, rewrites);
  EXPECT_EQ(result, ACL_SUCCESS);
  ASSERT_EQ(touched.size(), tasks.size());
  EXPECT_EQ(updateCount, 1U);
  for (size_t index = 0; index < tasks.size(); ++index) {
    EXPECT_EQ(touched[index], &tasks[index]);
    EXPECT_EQ(tasks[index].params.kernelTaskParams.funcHandle, rewrites[index]->specFuncHandle);
    auto restored = tasks[index].params;
    restored.kernelTaskParams.funcHandle = rewrites[index]->originalParams.kernelTaskParams.funcHandle;
    EXPECT_EQ(std::memcmp(&restored, &rewrites[index]->originalParams, sizeof(restored)), 0);
  }
}

TEST_F(AclSpecOptimizerUpdateTest, FailedSetStopsWithoutTouchingTheRest) {
  setResults = {ACL_SUCCESS, ACL_ERROR_FAILURE};
  const auto result = updater.UpdateTasks(model, rewrites);
  EXPECT_EQ(result, ACL_ERROR_FAILURE);
  // Nothing is written back, so the task that was already rewritten keeps its specialized handle.
  EXPECT_EQ(touched, (std::vector<aclmdlRITask>{&tasks[0], &tasks[1]}));
  EXPECT_EQ(updateCount, 0U);
  EXPECT_EQ(tasks[0].params.kernelTaskParams.funcHandle, rewrites[0]->specFuncHandle);
}

TEST_F(AclSpecOptimizerUpdateTest, FirstFailedSetStopsBeforeTheRest) {
  setResults = {ACL_ERROR_FAILURE};
  const auto result = updater.UpdateTasks(model, rewrites);
  EXPECT_EQ(result, ACL_ERROR_FAILURE);
  EXPECT_EQ(touched, (std::vector<aclmdlRITask>{&tasks[0]}));
  EXPECT_EQ(updateCount, 0U);
}

TEST_F(AclSpecOptimizerUpdateTest, FailedUpdateIsReportedOnce) {
  updateResults = {ACL_ERROR_FAILURE};
  const auto result = updater.UpdateTasks(model, rewrites);
  EXPECT_EQ(result, ACL_ERROR_FAILURE);
  EXPECT_EQ(touched.size(), tasks.size());
  EXPECT_EQ(updateCount, 1U);
}

TEST(SpecOutputDirectoryTest, SameModelGetsUniqueDirectoriesAndRestrictedPermissions) {
  SpecOutputDirectory first;
  SpecOutputDirectory second;
  ASSERT_EQ(first.Create(17), ACL_SUCCESS);
  ASSERT_EQ(second.Create(17), ACL_SUCCESS);
  EXPECT_NE(first.Path(), second.Path());
  struct stat info {};
  ASSERT_EQ(stat(first.Path().c_str(), &info), 0);
  EXPECT_EQ(info.st_mode & 0777, 0700);
  EXPECT_NE(first.Path().find("_17_"), std::string::npos);
  EXPECT_EQ(first.Create(17), ACL_ERROR_INVALID_PARAM);
}

TEST(SpecOutputDirectoryTest, CleanupRemovesOnlyOwnedDirectoryAndDoesNotFollowSymlinks) {
  SpecOutputDirectory neighbor;
  ASSERT_EQ(neighbor.Create(17), ACL_SUCCESS);
  const auto protectedFile = std::filesystem::path(neighbor.Path()) / "keep.elf";
  std::ofstream(protectedFile) << "keep";
  std::string removedPath;
  {
    SpecOutputDirectory current;
    ASSERT_EQ(current.Create(17), ACL_SUCCESS);
    removedPath = current.Path();
    std::ofstream(std::filesystem::path(removedPath) / "task.elf") << "temporary";
    std::filesystem::create_directory_symlink(neighbor.Path(), std::filesystem::path(removedPath) / "link");
  }
  EXPECT_FALSE(std::filesystem::exists(removedPath));
  EXPECT_TRUE(std::filesystem::exists(protectedFile));
}

TEST(SpecOutputDirectoryTest, UncreatedDirectoryHasNothingToClean) {
  SpecOutputDirectory directory;
  EXPECT_TRUE(directory.Path().empty());
}

TEST(SpecOutputDirectoryTest, InvalidTemporaryRootFailsWithoutDeletingExistingFiles) {
  SpecOutputDirectory parent;
  ASSERT_EQ(parent.Create(17), ACL_SUCCESS);
  const std::string existingFile = parent.Path() + "/not-a-directory";
  std::ofstream(existingFile) << "keep";
  const char *previous = std::getenv("TMPDIR");
  const bool hadPrevious = previous != nullptr;
  const std::string saved = hadPrevious ? previous : "";
  ASSERT_EQ(setenv("TMPDIR", existingFile.c_str(), 1), 0);
  SpecOutputDirectory directory;
  const aclError result = directory.Create(17);
  if (hadPrevious) {
    EXPECT_EQ(setenv("TMPDIR", saved.c_str(), 1), 0);
  } else {
    EXPECT_EQ(unsetenv("TMPDIR"), 0);
  }
  EXPECT_EQ(result, ACL_ERROR_FAILURE);
  EXPECT_TRUE(directory.Path().empty());
  EXPECT_TRUE(std::filesystem::exists(existingFile));
}

namespace {
struct MetaEntry {
  uint16_t type;
  uint16_t length;
  std::vector<uint8_t> value;
};

// {4, 4, feature} is the shape every asc-devkit feature header contributes to .ascend.meta:
// l2cache = 3, debug = 1, assert = 5 and print = 4. Each is eight bytes, so in a kernel built
// with any of them the resource ID is not the first entry of the section.
MetaEntry FeatureEntry(uint8_t feature) {
  return {4, 4, {feature, 0, 0, 0}};
}

MetaEntry ResourceIdEntry(char digit) {
  return {SPEC_RESOURCE_ID_TLV_TYPE, SPEC_RESOURCE_ID_TLV_LENGTH,
          std::vector<uint8_t>(SPEC_RESOURCE_ID_TLV_LENGTH, static_cast<uint8_t>(digit))};
}

std::vector<uint8_t> MakeMetaSection(const std::vector<MetaEntry> &entries) {
  std::vector<uint8_t> section;
  for (const auto &entry : entries) {
    const uint16_t head[]{entry.type, entry.length};
    const auto *headBytes = reinterpret_cast<const uint8_t *>(head);
    section.insert(section.end(), headBytes, headBytes + sizeof(head));
    section.insert(section.end(), entry.value.begin(), entry.value.end());
  }
  return section;
}

// Section 0 is null and section 1 is .shstrtab, so the sole metadata section of a default
// ELF stays at index 2 for the tests that rewrite it through ChangeSection.
std::vector<uint8_t> MakeElfWithMetaSections(const std::vector<std::vector<uint8_t>> &metaSections) {
  const char names[] = "\0.shstrtab\0.ascend.meta\0";
  const size_t sectionCount = 2 + metaSections.size();
  const size_t stringsOffset = sizeof(Elf64_Ehdr) + sectionCount * sizeof(Elf64_Shdr);
  std::vector<size_t> metaOffsets;
  size_t nextOffset = stringsOffset + sizeof(names);
  for (const auto &metaSection : metaSections) {
    metaOffsets.push_back(nextOffset);
    nextOffset += metaSection.size();
  }
  std::vector<uint8_t> bytes(nextOffset, 0);
  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, ELFMAG, SELFMAG);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_VERSION] = EV_CURRENT;
  header.e_version = EV_CURRENT;
  header.e_ehsize = sizeof(header);
  header.e_shoff = sizeof(header);
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = static_cast<Elf64_Half>(sectionCount);
  header.e_shstrndx = 1;
  std::memcpy(bytes.data(), &header, sizeof(header));
  std::vector<Elf64_Shdr> sections(sectionCount);
  sections[1].sh_name = 1;
  sections[1].sh_type = SHT_STRTAB;
  sections[1].sh_offset = stringsOffset;
  sections[1].sh_size = sizeof(names);
  for (size_t index = 0; index < metaSections.size(); ++index) {
    sections[2 + index].sh_name = 11;
    sections[2 + index].sh_type = SHT_NOTE;
    sections[2 + index].sh_offset = metaOffsets[index];
    sections[2 + index].sh_size = metaSections[index].size();
  }
  std::memcpy(bytes.data() + header.e_shoff, sections.data(), sectionCount * sizeof(Elf64_Shdr));
  std::memcpy(bytes.data() + stringsOffset, names, sizeof(names));
  for (size_t index = 0; index < metaSections.size(); ++index) {
    if (metaSections[index].empty()) {
      continue;
    }
    std::memcpy(bytes.data() + metaOffsets[index], metaSections[index].data(), metaSections[index].size());
  }
  return bytes;
}

std::vector<uint8_t> MakeElf() {
  return MakeElfWithMetaSections({MakeMetaSection({ResourceIdEntry('a')})});
}

void ChangeSection(std::vector<uint8_t> &bytes, size_t index, const Elf64_Shdr &section) {
  std::memcpy(bytes.data() + sizeof(Elf64_Ehdr) + index * sizeof(section), &section, sizeof(section));
}

Elf64_Shdr GetSection(const std::vector<uint8_t> &bytes, size_t index) {
  Elf64_Shdr section{};
  std::memcpy(&section, bytes.data() + sizeof(Elf64_Ehdr) + index * sizeof(section), sizeof(section));
  return section;
}
}  // namespace

TEST(SpecResourceIdTest, ReadsNonNullTerminatedResourceId) {
  const auto bytes = MakeElf();
  std::string result;
  ASSERT_TRUE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
  EXPECT_EQ(result, std::string(64, 'a'));
}

TEST(SpecResourceIdTest, RejectsEveryTruncatedPrefixWithoutChangingOutput) {
  const auto bytes = MakeElf();
  for (size_t size = 0; size < bytes.size(); ++size) {
    std::string result = "unchanged";
    EXPECT_FALSE(ReadSpecResourceId(bytes.data(), size, result)) << size;
    EXPECT_EQ(result, "unchanged");
  }
}

TEST(SpecResourceIdTest, RejectsInvalidHeadersAndNullInput) {
  auto bytes = MakeElf();
  std::string result;
  EXPECT_FALSE(ReadSpecResourceId(nullptr, bytes.size(), result));
  bytes[EI_MAG0] = 0;
  EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
  bytes = MakeElf();
  bytes[EI_DATA] = ELFDATA2MSB;
  EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
  bytes = MakeElf();
  Elf64_Ehdr header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  header.e_shoff = std::numeric_limits<uint64_t>::max();
  std::memcpy(bytes.data(), &header, sizeof(header));
  EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
}

TEST(SpecResourceIdTest, RejectsInvalidSectionBoundsAndNames) {
  auto bytes = MakeElf();
  std::string result;
  auto section = GetSection(bytes, 2);
  section.sh_offset = std::numeric_limits<uint64_t>::max();
  ChangeSection(bytes, 2, section);
  EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
  bytes = MakeElf();
  section = GetSection(bytes, 2);
  section.sh_name = std::numeric_limits<uint32_t>::max();
  ChangeSection(bytes, 2, section);
  EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
  bytes = MakeElf();
  section = GetSection(bytes, 1);
  std::memset(bytes.data() + section.sh_offset, 'x', section.sh_size);
  EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
}

TEST(SpecResourceIdTest, RejectsMissingOrInvalidResourcePayload) {
  auto bytes = MakeElf();
  std::string result;
  bytes.back() = 'z';
  EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
  bytes = MakeElf();
  auto section = GetSection(bytes, 2);
  section.sh_size = 63;
  ChangeSection(bytes, 2, section);
  EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
  bytes = MakeElf();
  section = GetSection(bytes, 2);
  section.sh_type = SHT_NOBITS;
  ChangeSection(bytes, 2, section);
  EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
}

TEST(SpecResourceIdTest, WireFormatMatchesAscDevkitMetadataEntry) {
  // Pins what this repository hard-codes. The defining header lives in asc-devkit and is not
  // reachable from this build, so a rename or renumbering there cannot fail here; this only
  // catches an accidental local edit and records where the other half of the contract lives.
  EXPECT_EQ(SPEC_RESOURCE_ID_SECTION_NAME, ".ascend.meta");
  EXPECT_EQ(SPEC_RESOURCE_ID_TLV_TYPE, 6);
  EXPECT_EQ(SPEC_RESOURCE_ID_TLV_LENGTH, 64);
  // sizeof(BinaryMetaSpecializationResourceId): a two-field BaseTlv head plus the value.
  EXPECT_EQ(2 * sizeof(uint16_t) + SPEC_RESOURCE_ID_TLV_LENGTH, 68U);
  std::string result = "unchanged";
  for (const uint16_t neighbouringType : {uint16_t{5}, uint16_t{7}}) {
    auto entry = ResourceIdEntry('a');
    entry.type = neighbouringType;
    const auto bytes = MakeElfWithMetaSections({MakeMetaSection({entry})});
    EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result)) << neighbouringType;
  }
  auto shortEntry = ResourceIdEntry('a');
  shortEntry.length = SPEC_RESOURCE_ID_TLV_LENGTH - 1;
  shortEntry.value.pop_back();
  const auto bytes = MakeElfWithMetaSections({MakeMetaSection({shortEntry})});
  EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
  EXPECT_EQ(result, "unchanged");
}

TEST(SpecResourceIdTest, ReadsResourceIdAmongTheFeatureEntries) {
  // A kernel's .ascend.meta is the concatenation of every contributing header's entry, so the
  // resource ID is neither the first nor necessarily the last entry of the section.
  const std::vector<std::vector<MetaEntry>> layouts{
      {FeatureEntry(3), FeatureEntry(1), FeatureEntry(5), ResourceIdEntry('a')},
      {FeatureEntry(3), ResourceIdEntry('a'), FeatureEntry(4)},
  };
  for (const auto &entries : layouts) {
    const auto bytes = MakeElfWithMetaSections({MakeMetaSection(entries)});
    std::string result;
    ASSERT_TRUE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
    EXPECT_EQ(result, std::string(SPEC_RESOURCE_ID_TLV_LENGTH, 'a'));
  }
}

TEST(SpecResourceIdTest, RejectsRepeatedResourceIdWhereverItIsDuplicated) {
  std::string result = "unchanged";
  // ld.lld -r concatenates same-name entries rather than replacing them.
  const auto repeatedInOneSection =
      MakeElfWithMetaSections({MakeMetaSection({ResourceIdEntry('a'), ResourceIdEntry('b')})});
  EXPECT_FALSE(ReadSpecResourceId(repeatedInOneSection.data(), repeatedInOneSection.size(), result));
  // objcopy --add-section on an existing section appends a second section of the same name.
  const auto repeatedAcrossSections =
      MakeElfWithMetaSections({MakeMetaSection({ResourceIdEntry('a')}), MakeMetaSection({ResourceIdEntry('b')})});
  EXPECT_FALSE(ReadSpecResourceId(repeatedAcrossSections.data(), repeatedAcrossSections.size(), result));
  EXPECT_EQ(result, "unchanged");
  // A single entry stays unambiguous even when the section itself is duplicated.
  const auto singleAcrossSections =
      MakeElfWithMetaSections({MakeMetaSection({FeatureEntry(3)}), MakeMetaSection({ResourceIdEntry('b')})});
  ASSERT_TRUE(ReadSpecResourceId(singleAcrossSections.data(), singleAcrossSections.size(), result));
  EXPECT_EQ(result, std::string(SPEC_RESOURCE_ID_TLV_LENGTH, 'b'));
}

TEST(SpecResourceIdTest, RejectsSectionThatCarriesOnlyFeatureEntries) {
  std::string result = "unchanged";
  const auto bytes = MakeElfWithMetaSections({MakeMetaSection({FeatureEntry(3), FeatureEntry(1)})});
  EXPECT_FALSE(ReadSpecResourceId(bytes.data(), bytes.size(), result));
  EXPECT_EQ(result, "unchanged");
}

namespace {
struct KernelState {
  std::string name = "AddCustom";
  std::vector<uint8_t> binary;
  size_t paramCount = 3;
  size_t offset = 16;
  size_t size = 8;
  aclError paramError = ACL_SUCCESS;
  aclError nameError = ACL_SUCCESS;
  bool unterminatedName = false;
};

using CollectorTask = TaskState;

struct StreamState {
  std::vector<aclmdlRITask> tasks;
};

struct ModelState {
  std::vector<aclrtStream> streams;
};

size_t streamQueries = 0;
size_t taskQueries = 0;
size_t failStreamQuery = 0;
size_t failTaskQuery = 0;
size_t fewerStreams = 0;
size_t fewerTasks = 0;
size_t binaryQueries = 0;
size_t paramCountQueries = 0;
size_t nameQueries = 0;
size_t copyQueries = 0;
aclError enumerationError = ACL_ERROR_FAILURE;
size_t parameterQueries = 0;
size_t lastParameterIndex = 0;
std::vector<aclError> specializationResults;
std::vector<std::string> specializationPaths;
size_t loadCalls = 0;
size_t functionCalls = 0;
size_t unloadCalls = 0;
aclError loadResult = ACL_SUCCESS;
aclError functionResult = ACL_SUCCESS;
int staticBinaryToken = 0;
int staticFunctionToken = 0;

std::vector<uint8_t> MakeKernelBinary() {
  auto bytes = MakeElf();
  const auto resource = GetSection(bytes, 2);
  std::fill(bytes.begin() + resource.sh_offset + sizeof(uint16_t) * 2,
            bytes.begin() + resource.sh_offset + resource.sh_size, 'b');
  return bytes;
}

aclError GetStreams(aclmdlRI model, aclrtStream *streams, uint32_t *count);
aclError GetTasks(aclrtStream stream, aclmdlRITask *tasks, uint32_t *count);
aclError GetFunctionName(aclrtFuncHandle function, uint32_t capacity, char *name);
aclError GetFunctionBinary(aclrtFuncHandle function, aclrtBinHandle *binary);
rtError_t GetBinaryBuffer(rtBinHandle binary, int type, void **data, uint32_t *size);
aclError GetBinaryFunction(aclrtBinHandle binary, const char *name, aclrtFuncHandle *function);

aclError CopyArgs(void *dst, size_t capacity, const void *src, size_t size, aclrtMemcpyKind kind) {
  ++copyQueries;
  EXPECT_EQ(kind, ACL_MEMCPY_DEVICE_TO_HOST);
  if (SkUtFailAclrtMemcpy() || src == nullptr || dst == nullptr || size > capacity) {
    return ACL_ERROR_FAILURE;
  }
  std::memcpy(dst, src, size);
  return ACL_SUCCESS;
}

void UnloadBinary(aclrtBinHandle binary) {
  EXPECT_EQ(binary, &staticBinaryToken);
  ++unloadCalls;
}

// Arm the allocator only after the selected identity has been read by the real implementation.
// This excludes setup and mock-framework allocations from the injected collection failure.
template <typename T>
class ScopedCollectionAllocationFailure {
 public:
  explicit ScopedCollectionAllocationFailure(uint32_t taskIndex) : taskIndex_(taskIndex) {
    identityHook_ = HookIdentity();
    active_ = this;
  }
  ~ScopedCollectionAllocationFailure() {
    allocationHook_.reset();
    identityHook_.reset();
    active_ = nullptr;
  }
  size_t failures = 0;
  std::string outputDirectory;

 private:
  static std::unique_ptr<mockcpp::ApiHook> HookIdentity() {
    return std::make_unique<mockcpp::ApiHook>(mockcpp::Details::methodToAddr(&AclSpecOptimizer::ReadTaskIdentity),
                                              reinterpret_cast<const void *>(&ReadIdentityAndArm));
  }
  static aclError ReadIdentityAndArm(const AclSpecOptimizer *optimizer, AclSpecOptimizer::TaskIdentity &identity) {
    auto &failure = *active_;
    failure.identityHook_.reset();
    const aclError result = optimizer->ReadTaskIdentity(identity);
    if (result == ACL_SUCCESS && identity.record.taskIndex == failure.taskIndex_) {
      failure.outputDirectory = optimizer->outputDirectory_.Path();
      failure.allocationHook_ =
          std::make_unique<mockcpp::ApiHook>(mockcpp::Details::methodToAddr(&std::allocator<T>::allocate),
                                             reinterpret_cast<const void *>(&FailAllocation));
    } else {
      failure.identityHook_ = HookIdentity();
    }
    return result;
  }
  static T *FailAllocation(const void *, size_t, const void *) {
    ++active_->failures;
    // Restore before unwinding so exception logging and cleanup can allocate normally.
    active_->allocationHook_.reset();
    throw std::bad_alloc();
  }

  inline static ScopedCollectionAllocationFailure *active_ = nullptr;
  uint32_t taskIndex_;
  std::unique_ptr<mockcpp::ApiHook> identityHook_;
  std::unique_ptr<mockcpp::ApiHook> allocationHook_;
};

class AclSpecOptimizerCollectionTest : public testing::Test {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
    MOCKER(aclmdlRITaskSetParams).stubs().will(invoke(SetTaskParams));
    MOCKER(aclmdlRIUpdate).stubs().will(invoke(UpdateModel));
    MOCKER(aclmdlRIGetStreams).stubs().will(invoke(GetStreams));
    MOCKER(aclmdlRIGetTasksByStream).stubs().will(invoke(GetTasks));
    MOCKER(aclmdlRITaskGetParams).stubs().will(invoke(GetTaskParams));
    MOCKER(aclmdlRITaskGetType).stubs().will(invoke(GetTaskType));
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(GetFunctionName));
    MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(GetFunctionBinary));
    MOCKER(aclrtMemcpy).stubs().will(invoke(CopyArgs));
    MOCKER(rtGetBinBuffer).stubs().will(invoke(GetBinaryBuffer));
    MOCKER(aclrtBinaryGetFunction).stubs().will(invoke(GetBinaryFunction));
    MOCKER(KernelSpecRequest::UnloadBinary).stubs().will(invoke(UnloadBinary));
    streamQueries = taskQueries = failStreamQuery = failTaskQuery = parameterQueries = 0;
    fewerStreams = fewerTasks = binaryQueries = paramCountQueries = nameQueries = copyQueries = 0;
    identityParamsQueries = identityTypeQueries = 0;
    enumerationError = ACL_ERROR_FAILURE;
    lastParameterIndex = 0;
    specializationResults.clear();
    specializationPaths.clear();
    loadCalls = functionCalls = 0;
    unloadCalls = 0;
    loadResult = functionResult = ACL_SUCCESS;
    setResults.clear();
    updateResults.clear();
    touched.clear();
    updateCount = 0;
    kernel.binary = MakeKernelBinary();
    task.type = ACL_MODEL_RI_TASK_KERNEL;
    task.params.type = ACL_MODEL_RI_TASK_KERNEL;
    task.params.kernelTaskParams.funcHandle = &kernel;
    task.params.kernelTaskParams.args = args.data();
    task.params.kernelTaskParams.argsSize = args.size();
    task.params.kernelTaskParams.numBlocks = 7;
    // A captured kernel task keeps its arguments on the device.
    task.params.kernelTaskParams.isHostArgs = 0;
    stream.tasks = {&task};
    model.streams = {&stream};
    ASSERT_TRUE(manager.Init(nullptr));
  }

  void TearDown() override {
    if (SkUtGetModelDestroyCallbackCount() != 0) {
      EXPECT_EQ(SkUtInvokeModelDestroyCallback(&model), ACL_SUCCESS);
    }
    GlobalMockObject::verify();
    SkUtResetTestControls();
  }

  aclError Collect() {
    return collector.Collect(&model, manager, "/tmp/spec-output", requests, scopeMarkers);
  }
  CollectorTask MakeMarker(KernelState &markerKernel, const char *name) {
    markerKernel.name = name;
    CollectorTask marker = task;
    marker.params.kernelTaskParams.funcHandle = &markerKernel;
    marker.params.kernelTaskParams.args = nullptr;
    marker.params.kernelTaskParams.argsSize = 0;
    return marker;
  }
  template <typename T>
  void CheckCollectionAllocationFailure(uint32_t taskIndex) {
    ASSERT_TRUE(manager.Init(nullptr, true));
    KernelState beginKernel;
    KernelState endKernel;
    auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_2201);
    auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_2201);
    stream.tasks = {&begin, &task, &end};
    const auto originalBegin = begin.params;
    const auto originalEnd = end.params;
    const auto originalTask = task.params;
    auto previous = std::make_unique<KernelSpecRequest>();
    const auto *saved = previous.get();
    requests.push_back(std::move(previous));
    scopeMarkers = {&task};
    MOCKER(aclmdlRITaskDisable).expects(never());
    for (bool publicEntry : {false, true}) {
      identityParamsQueries = nameQueries = 0;
      size_t failures = 0;
      std::string outputDirectory;
      aclError result = ACL_SUCCESS;
      bool allocationThrown = false;
      {
        ScopedCollectionAllocationFailure<T> failure(taskIndex);
        if (publicEntry) {
          result = aclmdlRISpecOptimizeByFlag(&model, nullptr);
        } else {
          try {
            Collect();
          } catch (const std::bad_alloc &) {
            allocationThrown = true;
          }
        }
        failures = failure.failures;
        outputDirectory = failure.outputDirectory;
      }
      EXPECT_EQ(failures, 1U);
      EXPECT_EQ(identityParamsQueries, taskIndex + 1U);
      EXPECT_EQ(nameQueries, taskIndex + 1U);
      if (publicEntry) {
        EXPECT_EQ(result, ACL_ERROR_BAD_ALLOC);
        ASSERT_FALSE(outputDirectory.empty());
        EXPECT_FALSE(std::filesystem::exists(outputDirectory));
      } else {
        EXPECT_TRUE(allocationThrown);
      }
      ASSERT_EQ(requests.size(), 1U);
      EXPECT_EQ(requests.front().get(), saved);
      EXPECT_EQ(scopeMarkers, (std::vector<aclmdlRITask>{&task}));
      EXPECT_EQ(binaryQueries, 0U);
      EXPECT_EQ(paramCountQueries, 0U);
      EXPECT_EQ(copyQueries, 0U);
      EXPECT_TRUE(specializationPaths.empty());
      EXPECT_EQ(loadCalls, 0U);
      EXPECT_TRUE(touched.empty());
      EXPECT_EQ(updateCount, 0U);
      EXPECT_EQ(std::memcmp(&begin.params, &originalBegin, sizeof(originalBegin)), 0);
      EXPECT_EQ(std::memcmp(&end.params, &originalEnd, sizeof(originalEnd)), 0);
      EXPECT_EQ(std::memcmp(&task.params, &originalTask, sizeof(originalTask)), 0);
    }
    // The scoped hooks must leave the same collector and allocators usable by later tests.
    ASSERT_EQ(Collect(), ACL_SUCCESS);
    ASSERT_EQ(requests.size(), 1U);
    EXPECT_EQ(requests.front()->task, &task);
    EXPECT_EQ(scopeMarkers, (std::vector<aclmdlRITask>{&begin, &end}));
  }
  std::array<uint8_t, 128> args{};
  KernelState kernel;
  CollectorTask task{};
  StreamState stream;
  ModelState model;
  FeatureManager manager;
  AclSpecOptimizerTestAccess collector;
  std::vector<std::unique_ptr<KernelSpecRequest>> requests;
  std::vector<aclmdlRITask> scopeMarkers;
};
}  // namespace

extern "C" aclError aclrtcKernelSpecialization(const aclrtcKernelSpecRequest *request, const char *path) {
  EXPECT_NE(request->resourceId, nullptr);
  EXPECT_NE(request->kernelEntry, nullptr);
  EXPECT_EQ(request->argsCount, 3U);
  EXPECT_NE(request->argsAddr[0], nullptr);
  EXPECT_GT(request->argsBytes[0], 0U);
  EXPECT_TRUE(std::filesystem::is_directory(std::filesystem::path(path).parent_path()));
  const size_t index = specializationPaths.size();
  specializationPaths.emplace_back(path);
  return index < specializationResults.size() ? specializationResults[index] : ACL_SUCCESS;
}

extern "C" aclError aclrtBinaryLoadFromFile(const char *, aclrtBinaryLoadOptions *, aclrtBinHandle *binary) {
  ++loadCalls;
  *binary = loadResult == ACL_SUCCESS ? &staticBinaryToken : nullptr;
  return loadResult;
}

namespace {
aclError GetBinaryFunction(aclrtBinHandle, const char *, aclrtFuncHandle *function) {
  ++functionCalls;
  *function = &staticFunctionToken;
  return functionResult;
}

aclError GetStreams(aclmdlRI model, aclrtStream *streams, uint32_t *count) {
  ++streamQueries;
  if (streamQueries == failStreamQuery) {
    return enumerationError;
  }
  const auto &values = static_cast<ModelState *>(model)->streams;
  if (streams == nullptr) {
    *count = values.size();
    return ACL_SUCCESS;
  }
  const size_t available = values.size() - fewerStreams;
  const size_t copied = std::min<size_t>(*count, available);
  *count = copied;
  if (copied < available) {
    return ACL_ERROR_RT_INSUFFICIENT_INPUT_ARRAY;
  }
  std::copy_n(values.begin(), copied, streams);
  return ACL_SUCCESS;
}

aclError GetTasks(aclrtStream stream, aclmdlRITask *tasks, uint32_t *count) {
  ++taskQueries;
  if (taskQueries == failTaskQuery) {
    return enumerationError;
  }
  const auto &values = static_cast<StreamState *>(stream)->tasks;
  if (tasks == nullptr) {
    *count = values.size();
    return ACL_SUCCESS;
  }
  const size_t available = values.size() - fewerTasks;
  const size_t copied = std::min<size_t>(*count, available);
  std::copy_n(values.begin(), copied, tasks);
  std::fill(tasks + copied, tasks + *count, nullptr);
  *count = copied;
  return copied < available ? ACL_ERROR_RT_INSUFFICIENT_INPUT_ARRAY : ACL_SUCCESS;
}

// Deliberately violate the Runtime contract only in the defensive enumeration tests.
aclError GetStreamsWithInvalidCount(aclmdlRI model, aclrtStream *streams, uint32_t *count) {
  const auto result = GetStreams(model, streams, count);
  if (result == ACL_SUCCESS && streams != nullptr) {
    ++*count;
  }
  return result;
}

aclError GetTasksWithInvalidCount(aclrtStream stream, aclmdlRITask *tasks, uint32_t *count) {
  const auto result = GetTasks(stream, tasks, count);
  if (result == ACL_SUCCESS && tasks != nullptr) {
    ++*count;
  }
  return result;
}

aclError GetFunctionName(aclrtFuncHandle function, uint32_t capacity, char *name) {
  ++nameQueries;
  const auto &state = *static_cast<KernelState *>(function);
  if (state.nameError != ACL_SUCCESS) {
    return state.nameError;
  }
  if (state.unterminatedName) {
    std::memset(name, 'x', capacity);
    return ACL_SUCCESS;
  }
  const auto &value = state.name;
  if (value.size() >= capacity) {
    return ACL_ERROR_FAILURE;
  }
  std::memcpy(name, value.c_str(), value.size() + 1);
  return ACL_SUCCESS;
}

aclError GetFunctionBinary(aclrtFuncHandle function, aclrtBinHandle *binary) {
  ++binaryQueries;
  *binary = function;
  return ACL_SUCCESS;
}

rtError_t GetBinaryBuffer(rtBinHandle binary, int type, void **data, uint32_t *size) {
  if (type != RT_BIN_HOST_ADDR) {
    return ACL_ERROR_INVALID_PARAM;
  }
  auto &value = static_cast<KernelState *>(binary)->binary;
  *data = value.data();
  *size = value.size();
  return RT_ERROR_NONE;
}
}  // namespace

extern "C" aclError aclrtFunctionGetParamCount(const void *function, size_t *count) {
  ++paramCountQueries;
  *count = static_cast<const KernelState *>(function)->paramCount;
  return ACL_SUCCESS;
}

extern "C" aclError aclrtFunctionGetParamInfo(const void *function, size_t index, size_t *offset, size_t *size) {
  const auto &value = *static_cast<const KernelState *>(function);
  ++parameterQueries;
  lastParameterIndex = index;
  *offset = value.offset + index * value.size;
  *size = value.size;
  return value.paramError;
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagLateIdentityFailuresPreserveOutputsWithoutCandidateWork) {
  ASSERT_TRUE(manager.Init(nullptr, true));
  for (bool secondStream : {false, true}) {
    for (int failure = 0; failure < 8; ++failure) {
      CollectorTask invalid = task;
      KernelState invalidKernel = kernel;
      invalid.params.kernelTaskParams.funcHandle = &invalidKernel;
      aclmdlRITask invalidHandle = &invalid;
      aclError expected = ACL_ERROR_FAILURE;
      switch (failure) {
        case 0:
          invalidHandle = nullptr;
          break;
        case 1:
          invalid.typeError = expected = ACL_ERROR_INVALID_PARAM;
          break;
        case 2:
          invalid.paramsError = expected = ACL_ERROR_INVALID_PARAM;
          break;
        case 3:
          invalid.params.type = ACL_MODEL_RI_TASK_EVENT_WAIT;
          break;
        case 4:
          invalid.params.kernelTaskParams.funcHandle = nullptr;
          break;
        case 5:
          invalidKernel.nameError = expected = ACL_ERROR_INVALID_PARAM;
          break;
        case 6:
          invalidKernel.name.clear();
          break;
        case 7:
          invalidKernel.unterminatedName = true;
          break;
      }
      StreamState second{{invalidHandle}};
      stream.tasks = secondStream ? std::vector<aclmdlRITask>{&task} : std::vector<aclmdlRITask>{&task, invalidHandle};
      model.streams = secondStream ? std::vector<aclrtStream>{&stream, &second} : std::vector<aclrtStream>{&stream};
      requests.clear();
      auto previous = std::make_unique<KernelSpecRequest>();
      const auto *saved = previous.get();
      requests.push_back(std::move(previous));
      scopeMarkers = {&task};
      binaryQueries = paramCountQueries = copyQueries = 0;
      EXPECT_EQ(Collect(), expected) << failure << " secondStream=" << secondStream;
      ASSERT_EQ(requests.size(), 1U);
      EXPECT_EQ(requests.front().get(), saved);
      EXPECT_EQ(scopeMarkers, (std::vector<aclmdlRITask>{&task}));
      EXPECT_EQ(binaryQueries, 0U);
      EXPECT_EQ(paramCountQueries, 0U);
      EXPECT_EQ(copyQueries, 0U);
      EXPECT_TRUE(specializationPaths.empty());
      EXPECT_TRUE(touched.empty());
      EXPECT_EQ(updateCount, 0U);
    }
  }
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagLateIdentityFailureStopsOptimizationBeforeAnyRuntimeMutation) {
  CollectorTask invalid = task;
  invalid.paramsError = ACL_ERROR_INVALID_PARAM;
  StreamState second{{&invalid}};
  model.streams.push_back(&second);
  MOCKER(aclmdlRITaskDisable).expects(never());
  EXPECT_EQ(aclmdlRISpecOptimizeByFlag(&model, nullptr), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(binaryQueries, 0U);
  EXPECT_EQ(paramCountQueries, 0U);
  EXPECT_EQ(copyQueries, 0U);
  EXPECT_TRUE(specializationPaths.empty());
  EXPECT_EQ(loadCalls, 0U);
  EXPECT_TRUE(touched.empty());
  EXPECT_EQ(updateCount, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, OptimizeSkipsInvalidIdentityAndCollectsLaterValidTask) {
  for (int failure = 0; failure < 8; ++failure) {
    CollectorTask invalid = task;
    KernelState invalidKernel = kernel;
    invalid.params.kernelTaskParams.funcHandle = &invalidKernel;
    aclmdlRITask handle = &invalid;
    switch (failure) {
      case 0:
        handle = nullptr;
        break;
      case 1:
        invalid.typeError = ACL_ERROR_INVALID_PARAM;
        break;
      case 2:
        invalid.paramsError = ACL_ERROR_INVALID_PARAM;
        break;
      case 3:
        invalid.params.type = ACL_MODEL_RI_TASK_EVENT_WAIT;
        break;
      case 4:
        invalid.params.kernelTaskParams.funcHandle = nullptr;
        break;
      case 5:
        invalidKernel.nameError = ACL_ERROR_INVALID_PARAM;
        break;
      case 6:
        invalidKernel.name.clear();
        break;
      case 7:
        invalidKernel.unterminatedName = true;
        break;
    }
    stream.tasks = {handle, &task};
    ASSERT_EQ(Collect(), ACL_SUCCESS);
    ASSERT_EQ(requests.size(), 1U) << failure;
    EXPECT_EQ(requests.front()->task, &task);
    EXPECT_EQ(requests.front()->taskIndex, 1U);
  }
}

TEST_F(AclSpecOptimizerCollectionTest, AllExactMarkersAllowZeroArgsAndNeverReadCandidateResources) {
  for (bool byFlag : {false, true}) {
    ASSERT_TRUE(manager.Init(nullptr, byFlag));
    for (const auto &names : {std::pair{SPEC_SCOPE_BEGIN_DAV_2201, SPEC_SCOPE_END_DAV_2201},
                              std::pair{SPEC_SCOPE_BEGIN_DAV_3510, SPEC_SCOPE_END_DAV_3510}}) {
      KernelState beginKernel;
      KernelState endKernel;
      auto begin = MakeMarker(beginKernel, names.first);
      auto end = MakeMarker(endKernel, names.second);
      stream.tasks = {&begin, &end};
      ASSERT_EQ(Collect(), ACL_SUCCESS);
      EXPECT_TRUE(requests.empty());
      EXPECT_EQ(scopeMarkers, (std::vector<aclmdlRITask>{&begin, &end}));
      EXPECT_EQ(binaryQueries, 0U);
      EXPECT_EQ(paramCountQueries, 0U);
      EXPECT_EQ(copyQueries, 0U);
    }
  }
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagSelectsRangeUnionAndSameNameTaskInstances) {
  ASSERT_TRUE(manager.Init(nullptr, true));
  // B/E are markers, K is a kernel task, O is a non-kernel task. Expected indexes select instances.
  const std::vector<std::pair<std::string, std::vector<uint32_t>>> cases{
      {"KK", {}},           {"BE", {}},         {"BEBE", {}},        {"KBKEK", {2}},
      {"BKKKE", {1, 2, 3}}, {"BKEBKE", {1, 4}}, {"BKEKBKE", {1, 5}}, {"BBBKEKEKE", {3, 5, 7}},
      {"BOKOE", {2}},       {"BOE", {}},
  };
  for (const auto &entry : cases) {
    const auto &layout = entry.first;
    SCOPED_TRACE(layout);
    KernelState beginKernel;
    KernelState endKernel;
    const auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_2201);
    const auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_2201);
    std::vector<CollectorTask> states(layout.size(), task);
    std::vector<aclmdlRITask> expectedMarkers;
    stream.tasks.clear();
    for (size_t index = 0; index < layout.size(); ++index) {
      if (layout[index] == 'B' || layout[index] == 'E') {
        states[index] = layout[index] == 'B' ? begin : end;
        expectedMarkers.push_back(&states[index]);
      } else if (layout[index] == 'O') {
        states[index].type = ACL_MODEL_RI_TASK_EVENT_WAIT;
        states[index].paramsError = ACL_ERROR_INVALID_PARAM;
      }
      stream.tasks.push_back(&states[index]);
    }
    binaryQueries = paramCountQueries = copyQueries = 0;
    ASSERT_EQ(Collect(), ACL_SUCCESS);
    ASSERT_EQ(requests.size(), entry.second.size());
    for (size_t index = 0; index < requests.size(); ++index) {
      EXPECT_EQ(requests[index]->task, &states[entry.second[index]]);
      EXPECT_EQ(requests[index]->taskIndex, entry.second[index]);
      EXPECT_EQ(requests[index]->streamIndex, 0U);
    }
    EXPECT_EQ(scopeMarkers, expectedMarkers);
    EXPECT_EQ(binaryQueries, requests.size());
    EXPECT_EQ(paramCountQueries, requests.size());
    EXPECT_EQ(copyQueries, requests.size());
  }
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagSupportsDeepNestingWithoutDuplicateRequests) {
  ASSERT_TRUE(manager.Init(nullptr, true));
  KernelState beginKernel;
  KernelState endKernel;
  auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_3510);
  auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_3510);
  for (size_t depth : {32U, 64U}) {
    // Distinct Runtime task instances may share the same marker function.
    std::vector<CollectorTask> begins(depth, begin);
    std::vector<CollectorTask> ends(depth, end);
    stream.tasks.clear();
    for (auto &marker : begins) {
      stream.tasks.push_back(&marker);
    }
    stream.tasks.push_back(&task);
    for (auto &marker : ends) {
      stream.tasks.push_back(&marker);
    }
    binaryQueries = copyQueries = 0;
    ASSERT_EQ(Collect(), ACL_SUCCESS);
    ASSERT_EQ(requests.size(), 1U);
    EXPECT_EQ(requests.front()->task, &task);
    EXPECT_EQ(requests.front()->taskIndex, depth);
    EXPECT_EQ(scopeMarkers.size(), depth * 2);
    EXPECT_EQ(binaryQueries, 1U);
    EXPECT_EQ(copyQueries, 1U);
  }
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagKeepsStreamRangesIndependentOfEnumerationOrder) {
  ASSERT_TRUE(manager.Init(nullptr, true));
  KernelState beginKernel;
  KernelState endKernel;
  auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_2201);
  auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_2201);
  auto secondBegin = begin;
  auto secondEnd = end;
  CollectorTask secondTask = task;
  CollectorTask outside = task;
  StreamState second{{&secondBegin, &secondTask, &secondEnd}};
  StreamState noScope{{&outside}};
  StreamState empty;
  stream.tasks = {&begin, &task, &end};
  for (bool reversed : {false, true}) {
    model.streams = reversed ? std::vector<aclrtStream>{&second, &empty, &noScope, &stream}
                             : std::vector<aclrtStream>{&stream, &noScope, &empty, &second};
    binaryQueries = copyQueries = 0;
    ASSERT_EQ(Collect(), ACL_SUCCESS);
    ASSERT_EQ(requests.size(), 2U);
    EXPECT_EQ(requests[0]->task, reversed ? &secondTask : &task);
    EXPECT_EQ(requests[1]->task, reversed ? &task : &secondTask);
    EXPECT_EQ(requests[0]->streamIndex, 0U);
    EXPECT_EQ(requests[1]->streamIndex, 3U);
    EXPECT_EQ(requests[0]->taskIndex, 1U);
    EXPECT_EQ(requests[1]->taskIndex, 1U);
    EXPECT_EQ(binaryQueries, 2U);
    EXPECT_EQ(copyQueries, 2U);
  }
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagMalformedRangesPreserveOutputsBeforeAnyCandidateWork) {
  ASSERT_TRUE(manager.Init(nullptr, true));
  const std::vector<std::pair<std::string, std::string>> cases{
      {"E", ""}, {"BK", ""}, {"BBKE", ""}, {"BKEE", ""}, {"BKEBK", ""}, {"BKE", "E"}, {"BKE", "BK"}, {"BK", "E"},
  };
  for (const auto &entry : cases) {
    SCOPED_TRACE(entry.first + "/" + entry.second);
    KernelState beginKernel;
    KernelState endKernel;
    const auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_2201);
    const auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_2201);
    std::vector<CollectorTask> firstStates(entry.first.size(), task);
    std::vector<CollectorTask> secondStates(entry.second.size(), task);
    StreamState second;
    auto fill = [&](const std::string &layout, std::vector<CollectorTask> &states, StreamState &target) {
      target.tasks.clear();
      for (size_t index = 0; index < layout.size(); ++index) {
        if (layout[index] == 'B' || layout[index] == 'E') {
          states[index] = layout[index] == 'B' ? begin : end;
        }
        target.tasks.push_back(&states[index]);
      }
    };
    fill(entry.first, firstStates, stream);
    fill(entry.second, secondStates, second);
    model.streams = {&stream, &second};
    requests.clear();
    auto previous = std::make_unique<KernelSpecRequest>();
    const auto *saved = previous.get();
    requests.push_back(std::move(previous));
    scopeMarkers = {&task};
    binaryQueries = paramCountQueries = copyQueries = 0;
    EXPECT_EQ(Collect(), ACL_ERROR_INVALID_PARAM);
    ASSERT_EQ(requests.size(), 1U);
    EXPECT_EQ(requests.front().get(), saved);
    EXPECT_EQ(scopeMarkers, (std::vector<aclmdlRITask>{&task}));
    EXPECT_EQ(binaryQueries, 0U);
    EXPECT_EQ(paramCountQueries, 0U);
    EXPECT_EQ(copyQueries, 0U);
  }
}

TEST_F(AclSpecOptimizerCollectionTest, OptimizeAdmitsAllOrdinaryKernelsWithoutPairValidation) {
  KernelState beginKernel;
  KernelState endKernel;
  auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_2201);
  auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_2201);
  CollectorTask secondTask = task;
  stream.tasks = {&end, &task, &begin, &secondTask};
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  ASSERT_EQ(requests.size(), 2U);
  EXPECT_EQ(requests[0]->task, &task);
  EXPECT_EQ(requests[1]->task, &secondTask);
  EXPECT_EQ(scopeMarkers, (std::vector<aclmdlRITask>{&end, &begin}));
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagLateRangeFailureStopsPublicEntryBeforeAnyRuntimeMutation) {
  KernelState beginKernel;
  KernelState endKernel;
  auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_2201);
  auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_2201);
  stream.tasks = {&begin, &task, &end};
  StreamState invalid{{&end}};
  model.streams.push_back(&invalid);
  MOCKER(aclmdlRITaskDisable).expects(never());
  EXPECT_EQ(aclmdlRISpecOptimizeByFlag(&model, nullptr), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(binaryQueries, 0U);
  EXPECT_EQ(paramCountQueries, 0U);
  EXPECT_EQ(copyQueries, 0U);
  EXPECT_TRUE(specializationPaths.empty());
  EXPECT_EQ(loadCalls, 0U);
  EXPECT_TRUE(touched.empty());
  EXPECT_EQ(updateCount, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, IdentityAllocationFailurePreservesOutputsAndMapsToBadAlloc) {
  CheckCollectionAllocationFailure<AclSpecOptimizer::TaskIdentity>(2);
}

TEST_F(AclSpecOptimizerCollectionTest, MarkerAllocationFailurePreservesOutputsAndMapsToBadAlloc) {
  CheckCollectionAllocationFailure<aclmdlRITask>(2);
}

TEST_F(AclSpecOptimizerCollectionTest, ActiveBeginAllocationFailurePreservesOutputsAndMapsToBadAlloc) {
  CheckCollectionAllocationFailure<size_t>(2);
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagNoScopeReturnsWithoutCandidateWorkOrMutation) {
  MOCKER(aclmdlRITaskDisable).expects(never());
  EXPECT_EQ(aclmdlRISpecOptimizeByFlag(&model, nullptr), ACL_SUCCESS);
  EXPECT_EQ(binaryQueries, 0U);
  EXPECT_EQ(paramCountQueries, 0U);
  EXPECT_EQ(copyQueries, 0U);
  EXPECT_TRUE(specializationPaths.empty());
  EXPECT_EQ(loadCalls, 0U);
  EXPECT_TRUE(touched.empty());
  EXPECT_EQ(updateCount, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagOnlyNonKernelInScopeReturnsWithoutCandidateWorkOrMutation) {
  KernelState beginKernel;
  KernelState endKernel;
  auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_3510);
  auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_3510);
  task.type = ACL_MODEL_RI_TASK_EVENT_RECORD;
  task.paramsError = ACL_ERROR_INVALID_PARAM;
  stream.tasks = {&begin, &task, &end};
  MOCKER(aclmdlRITaskDisable).expects(never());
  EXPECT_EQ(aclmdlRISpecOptimizeByFlag(&model, nullptr), ACL_SUCCESS);
  EXPECT_EQ(binaryQueries, 0U);
  EXPECT_EQ(paramCountQueries, 0U);
  EXPECT_EQ(copyQueries, 0U);
  EXPECT_TRUE(specializationPaths.empty());
  EXPECT_EQ(loadCalls, 0U);
  EXPECT_TRUE(touched.empty());
  EXPECT_EQ(updateCount, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagOnlySelectedTaskReadsResourcesCompilesAndRewrites) {
  KernelState beginKernel;
  KernelState endKernel;
  auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_2201);
  auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_2201);
  CollectorTask before = task;
  CollectorTask after = task;
  stream.tasks = {&before, &begin, &task, &end, &after};
  const auto originalParams = task.params;
  MOCKER(aclmdlRITaskDisable).expects(never());
  ASSERT_EQ(aclmdlRISpecOptimizeByFlag(&model, nullptr), ACL_SUCCESS);
  EXPECT_EQ(binaryQueries, 1U);
  EXPECT_EQ(paramCountQueries, 1U);
  EXPECT_EQ(copyQueries, 1U);
  ASSERT_EQ(specializationPaths.size(), 1U);
  EXPECT_NE(specializationPaths.front().find("stream_0_task_2.elf"), std::string::npos);
  EXPECT_EQ(loadCalls, 1U);
  EXPECT_EQ(touched, (std::vector<aclmdlRITask>{&task}));
  EXPECT_EQ(updateCount, 1U);
  EXPECT_EQ(std::memcmp(&before.params, &originalParams, sizeof(originalParams)), 0);
  EXPECT_EQ(std::memcmp(&after.params, &originalParams, sizeof(originalParams)), 0);
  EXPECT_EQ(begin.params.kernelTaskParams.funcHandle, &beginKernel);
  EXPECT_EQ(end.params.kernelTaskParams.funcHandle, &endKernel);
}

TEST_F(AclSpecOptimizerCollectionTest, SimilarAndSuperKernelNamesRemainOrdinaryKernelCandidates) {
  for (const char *name : {"spec_scope_kernel_begin", "spec_scope_kernel_begin_dav_2201_extra",
                           "spec_scope_kernel_end_dav_3510_extra", "sk_scope_kernel_begin_dav_2201"}) {
    kernel.name = name;
    ASSERT_EQ(Collect(), ACL_SUCCESS);
    ASSERT_EQ(requests.size(), 1U);
    EXPECT_EQ(requests.front()->kernelEntry, name);
    EXPECT_TRUE(scopeMarkers.empty());
  }
}

TEST_F(AclSpecOptimizerCollectionTest, NonKernelHasNoParamsAndIdentityIsReadOnceBeforeCandidateWork) {
  CollectorTask other = task;
  other.type = ACL_MODEL_RI_TASK_DEFAULT;
  other.paramsError = ACL_ERROR_INVALID_PARAM;
  stream.tasks = {&other, &task};
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  ASSERT_EQ(requests.size(), 1U);
  EXPECT_EQ(identityTypeQueries, 2U);
  EXPECT_EQ(identityParamsQueries, 1U);
  EXPECT_EQ(nameQueries, 1U);
  EXPECT_EQ(binaryQueries, 1U);
}

TEST_F(AclSpecOptimizerCollectionTest, ShrinkingEnumerationOnlyScansReturnedStreamsAndTasks) {
  model.streams.push_back(nullptr);
  fewerStreams = 1;
  stream.tasks.push_back(nullptr);
  fewerTasks = 1;
  ASSERT_TRUE(manager.Init(nullptr, true));
  task.type = ACL_MODEL_RI_TASK_DEFAULT;
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  EXPECT_EQ(taskQueries, 2U);
  EXPECT_EQ(identityTypeQueries, 1U);
  EXPECT_EQ(identityParamsQueries, 0U);
  EXPECT_TRUE(requests.empty());
}

TEST_F(AclSpecOptimizerCollectionTest, LateEnumerationErrorsPreserveOriginalCodeAndAvoidCandidateWork) {
  StreamState second{{&task}};
  model.streams.push_back(&second);
  enumerationError = ACL_ERROR_INVALID_PARAM;
  for (size_t failure : {3U, 4U}) {
    taskQueries = 0;
    failTaskQuery = failure;
    binaryQueries = copyQueries = 0;
    EXPECT_EQ(Collect(), ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(binaryQueries, 0U);
    EXPECT_EQ(copyQueries, 0U);
  }
  failTaskQuery = 0;
  for (size_t failure : {1U, 2U}) {
    streamQueries = 0;
    failStreamQuery = failure;
    EXPECT_EQ(Collect(), ACL_ERROR_INVALID_PARAM);
  }
  failStreamQuery = 0;
  model.streams[1] = nullptr;
  EXPECT_EQ(Collect(), ACL_ERROR_FAILURE);
  EXPECT_EQ(binaryQueries, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, UsesFeatureQueriesAndPassesRuntimeSizeWithoutDereferencing) {
  kernel.size = 24;
  // A null kernelName is the global rule, so it matches whatever entry the task reports.
  const char *basicArguments[] = {"-mllvm", "value with spaces", "-mllvm"};
  aclmdlRISpecCompileOption basicRule{nullptr, 3, basicArguments};
  aclmdlRISpecOptions options{};
  options.specCompileOptionCount = 1;
  options.specCompileOptions = &basicRule;
  options.enableSK = true;
  ASSERT_TRUE(manager.Init(&options));
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  ASSERT_EQ(requests.size(), 1U);
  auto &request = *requests.front();
  // Collection leaves the API request unbound; SpecializeKernels binds it right before use.
  request.BindApiRequest();
  EXPECT_EQ(request.task, &task);
  EXPECT_EQ(request.kernelEntry, kernel.name);
  EXPECT_EQ(request.resourceId, std::string(64, 'b'));
  EXPECT_EQ(request.argumentAddresses[0], request.hostArgs.data() + 16);
  ASSERT_EQ(request.argumentAddresses.size(), kernel.paramCount);
  ASSERT_EQ(request.argumentBytes.size(), kernel.paramCount);
  EXPECT_EQ(parameterQueries, kernel.paramCount);
  for (size_t index = 0; index < kernel.paramCount; ++index) {
    EXPECT_EQ(request.argumentAddresses[index], request.hostArgs.data() + kernel.offset + index * kernel.size);
    EXPECT_EQ(request.argumentBytes[index], kernel.size);
  }
  EXPECT_EQ(request.apiRequest.argsAddr, request.argumentAddresses.data());
  EXPECT_EQ(request.apiRequest.argsBytes, request.argumentBytes.data());
  EXPECT_EQ(request.apiRequest.resourceId, request.resourceId.c_str());
  EXPECT_EQ(request.apiRequest.kernelEntry, request.kernelEntry.c_str());
  EXPECT_EQ(request.apiRequest.options, request.basicOptionPointers.data());
  EXPECT_EQ(request.apiRequest.optionCount, request.basicOptionPointers.size());
  EXPECT_EQ(request.apiRequest.skOptions, request.skOptionPointers.data());
  EXPECT_EQ(request.apiRequest.skOptionCount, request.skOptionPointers.size());
  EXPECT_EQ(lastParameterIndex, kernel.paramCount - 1);
  EXPECT_EQ(request.basicOptionStorage, (std::vector<std::string>{"-mllvm", "value with spaces", "-mllvm"}));
  EXPECT_EQ(request.skOptionStorage, (std::vector<std::string>{"--enable-super-kernel"}));
  ASSERT_EQ(request.basicOptionPointers.size(), 3U);
  EXPECT_STREQ(request.basicOptionPointers[1], "value with spaces");
  EXPECT_STREQ(request.skOptionPointers[0], "--enable-super-kernel");
  EXPECT_EQ(request.outputElfPath, "/tmp/spec-output/stream_0_task_0.elf");
  EXPECT_EQ(task.params.kernelTaskParams.funcHandle, &kernel);
}

TEST_F(AclSpecOptimizerCollectionTest, OptimizerCompilesInOrderAndCommitsOnlySuccessfulTasks) {
  CollectorTask secondTask = task;
  stream.tasks = {&task, &secondTask};
  specializationResults = {ACL_ERROR_FAILURE, ACL_SUCCESS};
  const auto original = task.params;
  const auto result = collector.Optimize(&model, manager);
  EXPECT_EQ(result, ACL_SUCCESS);
  ASSERT_EQ(specializationPaths.size(), 2U);
  EXPECT_EQ(std::filesystem::path(specializationPaths[0]).filename(), "stream_0_task_0.elf");
  EXPECT_EQ(std::filesystem::path(specializationPaths[1]).filename(), "stream_0_task_1.elf");
  EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(specializationPaths[0]).parent_path()));
  EXPECT_EQ(loadCalls, 1U);
  EXPECT_EQ(functionCalls, 1U);
  EXPECT_EQ(updateCount, 1U);
  EXPECT_EQ(touched, (std::vector<aclmdlRITask>{&secondTask}));
  EXPECT_EQ(task.params.kernelTaskParams.funcHandle, original.kernelTaskParams.funcHandle);
  EXPECT_EQ(secondTask.params.kernelTaskParams.funcHandle, &staticFunctionToken);
  EXPECT_EQ(secondTask.params.kernelTaskParams.args, original.kernelTaskParams.args);
  EXPECT_EQ(secondTask.params.kernelTaskParams.numBlocks, original.kernelTaskParams.numBlocks);
}

// A kernel ACLRTC declines to specialize is skipped and reported as success; a binary that cannot
// be loaded, or whose entry is missing, aborts the whole model.
TEST_F(AclSpecOptimizerCollectionTest, SkipsCompileFailureButReportsLoadAndLookupFailures) {
  for (size_t stage = 0; stage < 3; ++stage) {
    specializationResults = {stage == 0 ? ACL_ERROR_FAILURE : ACL_SUCCESS};
    specializationPaths.clear();
    loadCalls = functionCalls = 0;
    loadResult = stage == 1 ? ACL_ERROR_FAILURE : ACL_SUCCESS;
    functionResult = stage == 2 ? ACL_ERROR_FAILURE : ACL_SUCCESS;
    const auto result = collector.Optimize(&model, manager);
    EXPECT_EQ(result, stage == 0 ? ACL_SUCCESS : ACL_ERROR_FAILURE);
    // No task is touched either way: loading happens before the model is modified.
    EXPECT_TRUE(touched.empty());
    // Stage 0 specializes nothing, so it reaches the commit with an empty list and spends one
    // no-op update on it. Stages 1 and 2 fail while loading and never get that far, so this
    // running count stays where stage 0 left it.
    EXPECT_EQ(updateCount, 1U);
    EXPECT_EQ(loadCalls, stage == 0 ? 0U : 1U);
    EXPECT_EQ(functionCalls, stage == 2 ? 1U : 0U);
    EXPECT_EQ(unloadCalls, stage == 2 ? 1U : 0U);
    EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 0U);
    ASSERT_EQ(specializationPaths.size(), 1U);
    EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(specializationPaths[0]).parent_path()));
  }
}

TEST_F(AclSpecOptimizerCollectionTest, OptimizerPropagatesCommitFailure) {
  setResults = {ACL_ERROR_FAILURE};
  const auto result = collector.Optimize(&model, manager);
  EXPECT_EQ(result, ACL_ERROR_FAILURE);
  EXPECT_EQ(touched.size(), 1U);
  EXPECT_EQ(updateCount, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, PublicEntryUsesDefaultOptionsAndCommits) {
  EXPECT_EQ(aclmdlRISpecOptimize(&model, nullptr), ACL_SUCCESS);
  EXPECT_EQ(task.params.kernelTaskParams.funcHandle, &staticFunctionToken);
  EXPECT_EQ(updateCount, 1U);
  EXPECT_EQ(unloadCalls, 0U);
  EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 1U);
  EXPECT_EQ(SkUtInvokeModelDestroyCallback(&model), ACL_SUCCESS);
  EXPECT_EQ(unloadCalls, 1U);
  ASSERT_EQ(specializationPaths.size(), 1U);
  EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(specializationPaths[0]).parent_path()));
}

TEST_F(AclSpecOptimizerCollectionTest, PublicEntryRejectsInvalidConfigurationBeforeTouchingTheModel) {
  // A non-zero rule count without a rule array is what the FeatureManager rejects.
  aclmdlRISpecOptions options{};
  options.specCompileOptionCount = 1;
  options.specCompileOptions = nullptr;
  EXPECT_EQ(aclmdlRISpecOptimize(nullptr, &options), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(aclmdlRISpecOptimize(&model, &options), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(streamQueries, 0U);
  EXPECT_TRUE(specializationPaths.empty());
  EXPECT_TRUE(touched.empty());
}

TEST_F(AclSpecOptimizerCollectionTest, PublicEntryPropagatesCommitFailureAndKeepsBinariesWithTheModel) {
  setResults = {ACL_ERROR_FAILURE};
  EXPECT_EQ(aclmdlRISpecOptimize(&model, nullptr), ACL_ERROR_FAILURE);
  EXPECT_EQ(touched.size(), 1U);
  EXPECT_EQ(updateCount, 0U);
  // A failed commit is not rolled back, so the binary stays loaded until the model is destroyed.
  EXPECT_EQ(unloadCalls, 0U);
  EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 1U);
  EXPECT_EQ(SkUtInvokeModelDestroyCallback(&model), ACL_SUCCESS);
  EXPECT_EQ(unloadCalls, 1U);
}

TEST_F(AclSpecOptimizerCollectionTest, FailedUpdateKeepsBinariesWithTheModel) {
  updateResults = {ACL_ERROR_FAILURE};
  EXPECT_EQ(collector.Optimize(&model, manager), ACL_ERROR_FAILURE);
  EXPECT_EQ(task.params.kernelTaskParams.funcHandle, &staticFunctionToken);
  EXPECT_EQ(updateCount, 1U);
  EXPECT_EQ(unloadCalls, 0U);
  EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 1U);
  EXPECT_EQ(SkUtInvokeModelDestroyCallback(&model), ACL_SUCCESS);
  EXPECT_EQ(unloadCalls, 1U);
}

TEST_F(AclSpecOptimizerCollectionTest, CallbackRegistrationFailureLeaksInsteadOfDanglingTheModel) {
  SkUtSetAclmdlRIDestroyRegisterCallbackRet(ACL_ERROR_FAILURE);
  EXPECT_EQ(collector.Optimize(&model, manager), ACL_ERROR_FAILURE);
  // The commit already switched the task over, so the binary has to stay loaded even though
  // nothing will ever release it.
  EXPECT_EQ(task.params.kernelTaskParams.funcHandle, &staticFunctionToken);
  EXPECT_EQ(touched.size(), 1U);
  EXPECT_EQ(updateCount, 1U);
  EXPECT_EQ(unloadCalls, 0U);
  EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, RepeatedOptimizationRegistersOneCallbackAndRetainsBothBinaries) {
  EXPECT_EQ(collector.Optimize(&model, manager), ACL_SUCCESS);
  CollectorTask another = task;
  another.params.kernelTaskParams.funcHandle = &kernel;
  stream.tasks = {&another};
  EXPECT_EQ(collector.Optimize(&model, manager), ACL_SUCCESS);
  EXPECT_EQ(unloadCalls, 0U);
  EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 1U);
  EXPECT_EQ(SkUtInvokeModelDestroyCallback(&model), ACL_SUCCESS);
  EXPECT_EQ(unloadCalls, 2U);
}

TEST_F(AclSpecOptimizerCollectionTest, NonKernelTaskIsSkipped) {
  task.type = ACL_MODEL_RI_TASK_EVENT_WAIT;
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  EXPECT_TRUE(requests.empty());
}

TEST_F(AclSpecOptimizerCollectionTest, NullTaskDoesNotPreventCollectingLaterKernel) {
  stream.tasks = {nullptr, &task};
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  ASSERT_EQ(requests.size(), 1U);
  EXPECT_EQ(requests[0]->task, &task);
  EXPECT_EQ(requests[0]->taskIndex, 1U);
}

TEST_F(AclSpecOptimizerCollectionTest, TypeQueryFailureDoesNotPreventCollectingLaterKernel) {
  CollectorTask failedTask{};
  failedTask.typeError = ACL_ERROR_FAILURE;
  stream.tasks = {&failedTask, &task};
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  ASSERT_EQ(requests.size(), 1U);
  EXPECT_EQ(requests[0]->task, &task);
  EXPECT_EQ(requests[0]->taskIndex, 1U);
}

TEST_F(AclSpecOptimizerCollectionTest, InvalidIdentityIsTaskLocal) {
  kernel.name.clear();
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  EXPECT_TRUE(requests.empty());
  kernel.name = "AddCustom";
  kernel.binary.clear();
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  EXPECT_TRUE(requests.empty());
}

TEST_F(AclSpecOptimizerCollectionTest, InvalidParameterRangeIsTaskLocal) {
  kernel.offset = std::numeric_limits<size_t>::max();
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  EXPECT_TRUE(requests.empty());
  kernel.offset = 16;
  kernel.size = std::numeric_limits<size_t>::max();
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  EXPECT_TRUE(requests.empty());
  kernel.size = 8;
  kernel.paramCount = 0;
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  EXPECT_TRUE(requests.empty());
}

TEST_F(AclSpecOptimizerCollectionTest, PartialFailuresKeepOriginalTaskOrderAndUniquePaths) {
  CollectorTask skipped = task;
  skipped.type = ACL_MODEL_RI_TASK_EVENT_WAIT;
  CollectorTask other = task;
  StreamState second{{&other}};
  stream.tasks = {&skipped, &task};
  model.streams.push_back(&second);
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  ASSERT_EQ(requests.size(), 2U);
  EXPECT_EQ(requests[0]->streamIndex, 0U);
  EXPECT_EQ(requests[0]->taskIndex, 1U);
  EXPECT_EQ(requests[1]->streamIndex, 1U);
  EXPECT_EQ(requests[1]->taskIndex, 0U);
  EXPECT_NE(requests[0]->outputElfPath, requests[1]->outputElfPath);
}

TEST_F(AclSpecOptimizerCollectionTest, EveryEnumerationFailureIsGlobal) {
  for (size_t failure = 1; failure <= 2; ++failure) {
    streamQueries = 0;
    failStreamQuery = failure;
    EXPECT_EQ(Collect(), ACL_ERROR_FAILURE);
    EXPECT_TRUE(requests.empty());
  }
  failStreamQuery = 0;
  for (size_t failure = 1; failure <= 2; ++failure) {
    taskQueries = 0;
    failTaskQuery = failure;
    EXPECT_EQ(Collect(), ACL_ERROR_FAILURE);
    EXPECT_TRUE(requests.empty());
  }
}

TEST_F(AclSpecOptimizerCollectionTest, EmptyModelDoesNotRepeatCountQueryWithNullStorage) {
  model.streams.clear();
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  EXPECT_EQ(streamQueries, 1U);
  EXPECT_EQ(taskQueries, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, RejectsStreamCountExceedingAllocatedCapacity) {
  GlobalMockObject::reset(reinterpret_cast<const void *>(aclmdlRIGetStreams));
  MOCKER(aclmdlRIGetStreams).stubs().will(invoke(GetStreamsWithInvalidCount));
  EXPECT_EQ(Collect(), ACL_ERROR_FAILURE);
  EXPECT_TRUE(requests.empty());
  EXPECT_EQ(taskQueries, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, RejectsTaskCountExceedingAllocatedCapacity) {
  GlobalMockObject::reset(reinterpret_cast<const void *>(aclmdlRIGetTasksByStream));
  MOCKER(aclmdlRIGetTasksByStream).stubs().will(invoke(GetTasksWithInvalidCount));
  EXPECT_EQ(Collect(), ACL_ERROR_FAILURE);
  EXPECT_TRUE(requests.empty());
  EXPECT_EQ(parameterQueries, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, StreamEnumerationStubRespectsArrayCapacity) {
  StreamState second;
  model.streams.push_back(&second);
  std::array<aclrtStream, 3> streams{&model, &model, &model};
  uint32_t count = 1;
  EXPECT_EQ(aclmdlRIGetStreams(&model, streams.data(), &count), ACL_ERROR_RT_INSUFFICIENT_INPUT_ARRAY);
  EXPECT_EQ(count, 1U);
  EXPECT_EQ(streams, (std::array<aclrtStream, 3>{&model, &model, &model}));

  count = streams.size();
  EXPECT_EQ(aclmdlRIGetStreams(&model, streams.data(), &count), ACL_SUCCESS);
  EXPECT_EQ(count, 2U);
  EXPECT_EQ(streams, (std::array<aclrtStream, 3>{&stream, &second, &model}));
}

TEST_F(AclSpecOptimizerCollectionTest, TaskEnumerationStubRespectsArrayCapacity) {
  CollectorTask second = task;
  stream.tasks.push_back(&second);
  std::array<aclmdlRITask, 3> tasks{&model, &model, &model};
  uint32_t count = 1;
  EXPECT_EQ(aclmdlRIGetTasksByStream(&stream, tasks.data(), &count), ACL_ERROR_RT_INSUFFICIENT_INPUT_ARRAY);
  EXPECT_EQ(count, 1U);
  EXPECT_EQ(tasks, (std::array<aclmdlRITask, 3>{&task, &model, &model}));

  count = tasks.size();
  EXPECT_EQ(aclmdlRIGetTasksByStream(&stream, tasks.data(), &count), ACL_SUCCESS);
  EXPECT_EQ(count, 2U);
  EXPECT_EQ(tasks, (std::array<aclmdlRITask, 3>{&task, &second, nullptr}));
}

TEST_F(AclSpecOptimizerCollectionTest, ModelWithoutMarkersClearsPreviousMarkers) {
  scopeMarkers.push_back(&task);
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  ASSERT_EQ(requests.size(), 1U);
  EXPECT_EQ(requests.front()->task, &task);
  EXPECT_TRUE(scopeMarkers.empty());
  EXPECT_EQ(task.params.kernelTaskParams.funcHandle, &kernel);
  EXPECT_TRUE(touched.empty());
  EXPECT_EQ(updateCount, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, RejectsInvalidInputsWithoutQueryingRuntime) {
  EXPECT_EQ(collector.Collect(nullptr, manager, "/tmp/spec-output", requests, scopeMarkers), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(collector.Collect(&model, manager, "", requests, scopeMarkers), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(streamQueries, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, PointerStorageRemainsValidAfterRequestListGrowth) {
  const std::string longArgument(200, 'x');
  const std::vector<std::string> expectedBasic{"-O2", "short", longArgument};
  const char *basicArguments[] = {"-O2", "short", longArgument.c_str()};
  aclmdlRISpecCompileOption basicRule{nullptr, 3, basicArguments};
  aclmdlRISpecOptions options{};
  options.specCompileOptionCount = 1;
  options.specCompileOptions = &basicRule;
  options.enableSK = true;
  ASSERT_TRUE(manager.Init(&options));
  stream.tasks.assign(129, &task);
  ASSERT_EQ(Collect(), ACL_SUCCESS);
  ASSERT_EQ(requests.size(), 129U);
  for (size_t index = 0; index < requests.size(); ++index) {
    auto &request = *requests[index];
    request.BindApiRequest();
    request.BindApiRequest();
    EXPECT_EQ(request.taskIndex, index);
    EXPECT_EQ(request.apiRequest.argsAddr, request.argumentAddresses.data());
    EXPECT_EQ(request.apiRequest.argsBytes, request.argumentBytes.data());
    EXPECT_EQ(request.apiRequest.resourceId, request.resourceId.c_str());
    EXPECT_EQ(request.apiRequest.kernelEntry, request.kernelEntry.c_str());
    EXPECT_EQ(request.apiRequest.options, request.basicOptionPointers.data());
    ASSERT_EQ(request.basicOptionPointers.size(), request.basicOptionStorage.size());
    for (size_t option = 0; option < request.basicOptionStorage.size(); ++option) {
      EXPECT_EQ(request.basicOptionPointers[option], request.basicOptionStorage[option].c_str());
      EXPECT_STREQ(request.basicOptionPointers[option], expectedBasic[option].c_str());
    }
  }
}

TEST_F(AclSpecOptimizerCollectionTest, GlobalFailureDoesNotPublishPreviouslyCollectedRequests) {
  StreamState second{{&task}};
  model.streams.push_back(&second);
  failTaskQuery = 3;
  EXPECT_EQ(Collect(), ACL_ERROR_FAILURE);
  EXPECT_TRUE(requests.empty());
}

TEST_F(AclSpecOptimizerCollectionTest, GlobalFailurePreservesBothPreviousOutputsAndBorrowedTask) {
  auto previous = std::make_unique<KernelSpecRequest>();
  previous->task = &task;
  const auto *previousRequest = previous.get();
  requests.push_back(std::move(previous));
  scopeMarkers.push_back(&task);
  StreamState second{{&task}};
  model.streams.push_back(&second);
  failTaskQuery = 3;
  EXPECT_EQ(Collect(), ACL_ERROR_FAILURE);
  ASSERT_EQ(requests.size(), 1U);
  EXPECT_EQ(requests.front().get(), previousRequest);
  EXPECT_EQ(requests.front()->task, &task);
  ASSERT_EQ(scopeMarkers.size(), 1U);
  EXPECT_EQ(scopeMarkers.front(), &task);
  EXPECT_EQ(task.params.kernelTaskParams.funcHandle, &kernel);
  EXPECT_TRUE(touched.empty());
  EXPECT_EQ(updateCount, 0U);
  EXPECT_EQ(unloadCalls, 0U);
}

TEST_F(AclSpecOptimizerCollectionTest, ZeroSizeNullArgsAndParameterQueryFailureAreTaskLocal) {
  kernel.size = 0;
  EXPECT_EQ(Collect(), ACL_SUCCESS);
  EXPECT_TRUE(requests.empty());
  kernel.size = 8;
  task.params.kernelTaskParams.args = nullptr;
  EXPECT_EQ(Collect(), ACL_SUCCESS);
  EXPECT_TRUE(requests.empty());
  task.params.kernelTaskParams.args = args.data();
  kernel.paramError = ACL_ERROR_FAILURE;
  EXPECT_EQ(Collect(), ACL_SUCCESS);
  EXPECT_TRUE(requests.empty());
}

namespace {
using OptimizeEntry = aclError (*)(aclmdlRI, aclmdlRISpecOptions *);
OptimizeEntry GetOptimizeEntry(bool isScopeJit) {
  return isScopeJit ? aclmdlRISpecOptimizeByFlag : aclmdlRISpecOptimize;
}

std::vector<std::pair<aclmdlRI, bool>> entryModes;
aclError InspectEntryMode(AclSpecOptimizer *, aclmdlRI model, const FeatureManager &manager) {
  entryModes.emplace_back(model, manager.Get<Feature::IS_SCOPE_JIT>());
  return ACL_SUCCESS;
}

class AclSpecOptimizerEntryTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
    entryModes.clear();
  }
  void TearDown() override {
    GlobalMockObject::verify();
    SkUtResetTestControls();
  }
};

TEST_P(AclSpecOptimizerEntryTest, PassesModeWithoutLeakingBetweenModels) {
  MOCKER_CPP(&AclSpecOptimizer::Optimize).stubs().will(invoke(InspectEntryMode));
  int first = 0;
  int second = 0;
  int third = 0;
  int fourth = 0;
  EXPECT_EQ(GetOptimizeEntry(GetParam())(&first, nullptr), ACL_SUCCESS);
  EXPECT_EQ(GetOptimizeEntry(!GetParam())(&second, nullptr), ACL_SUCCESS);
  EXPECT_EQ(GetOptimizeEntry(!GetParam())(&third, nullptr), ACL_SUCCESS);
  EXPECT_EQ(GetOptimizeEntry(GetParam())(&fourth, nullptr), ACL_SUCCESS);
  EXPECT_EQ(entryModes,
            (std::vector<std::pair<aclmdlRI, bool>>{
                {&first, GetParam()}, {&second, !GetParam()}, {&third, !GetParam()}, {&fourth, GetParam()}}));
}

TEST_P(AclSpecOptimizerEntryTest, RejectsConfigurationBeforeInitializingModel) {
  MOCKER_CPP(&AclSpecOptimizer::Init).expects(never());
  MOCKER_CPP(&AclSpecOptimizer::Optimize).expects(never());
  aclmdlRISpecOptions options{};
  options.specCompileOptionCount = 1;
  int model = 0;
  EXPECT_EQ(GetOptimizeEntry(GetParam())(nullptr, &options), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(GetOptimizeEntry(GetParam())(&model, &options), ACL_ERROR_INVALID_PARAM);
}

TEST_P(AclSpecOptimizerEntryTest, PropagatesInitializationErrorWithoutOptimizing) {
  MOCKER_CPP(&AclSpecOptimizer::Optimize).expects(never());
  SkUtSetAclmdlRIGetIdRet(ACL_ERROR_BAD_ALLOC);
  int model = 0;
  EXPECT_EQ(GetOptimizeEntry(GetParam())(&model, nullptr), ACL_ERROR_BAD_ALLOC);
}

TEST_P(AclSpecOptimizerEntryTest, RejectsNullModelWithValidConfiguration) {
  MOCKER_CPP(&AclSpecOptimizer::Optimize).expects(never());
  aclmdlRISpecOptions options{};
  EXPECT_EQ(GetOptimizeEntry(GetParam())(nullptr, &options), ACL_ERROR_INVALID_PARAM);
}

TEST_P(AclSpecOptimizerEntryTest, PropagatesOptimizationError) {
  MOCKER_CPP(&AclSpecOptimizer::Optimize).expects(once()).will(returnValue(ACL_ERROR_INVALID_PARAM));
  int model = 0;
  EXPECT_EQ(GetOptimizeEntry(GetParam())(&model, nullptr), ACL_ERROR_INVALID_PARAM);
}

INSTANTIATE_TEST_SUITE_P(PublicEntries, AclSpecOptimizerEntryTest, testing::Bool());

int entryException = 0;
aclError ThrowEntryException(const AclSpecOptimizer *, aclmdlRI, const FeatureManager &) {
  switch (entryException) {
    case 0:
      throw std::bad_alloc();
    case 1:
      throw std::length_error("entry test");
    case 2:
      throw std::invalid_argument("entry test");
    case 3:
      throw std::runtime_error("entry test");
    default:
      throw 1;
  }
}

aclError ThrowInitEntryException(AclSpecOptimizer *optimizer, aclmdlRI model) {
  return ThrowEntryException(optimizer, model, FeatureManager());
}

class AclSpecOptimizerEntryExceptionTest : public testing::TestWithParam<std::tuple<bool, int, bool>> {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
  }
  void TearDown() override {
    GlobalMockObject::verify();
    SkUtResetTestControls();
  }
};

TEST_P(AclSpecOptimizerEntryExceptionTest, ConvertsExceptionAtPublicBoundary) {
  entryException = std::get<1>(GetParam());
  // ChainableMockMethod catches std::exception and converts it to a mock failure.
  // A direct RAII hook preserves the exception type for the public API boundary.
  const bool failInit = std::get<2>(GetParam());
  if (failInit) {
    MOCKER_CPP(&AclSpecOptimizer::Optimize).expects(never());
  }
  mockcpp::ApiHook hook(failInit ? mockcpp::Details::methodToAddr(&AclSpecOptimizer::Init)
                                 : mockcpp::Details::methodToAddr(&AclSpecOptimizer::Optimize),
                        failInit ? reinterpret_cast<const void *>(&ThrowInitEntryException)
                                 : reinterpret_cast<const void *>(&ThrowEntryException));
  const std::array<aclError, 5> expected{ACL_ERROR_BAD_ALLOC, ACL_ERROR_BAD_ALLOC, ACL_ERROR_INVALID_PARAM,
                                         ACL_ERROR_FAILURE, ACL_ERROR_FAILURE};
  int model = 0;
  EXPECT_EQ(GetOptimizeEntry(std::get<0>(GetParam()))(&model, nullptr), expected[entryException]);
}

INSTANTIATE_TEST_SUITE_P(PublicEntries, AclSpecOptimizerEntryExceptionTest,
                         testing::Combine(testing::Bool(), testing::Range(0, 5), testing::Bool()));
}  // namespace

}  // namespace model_spec
