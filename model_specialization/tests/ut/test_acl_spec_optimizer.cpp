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
#include "dlog_pub.h"
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>
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
#include <functional>
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
                   std::vector<std::unique_ptr<KernelSpecRequest>> &requests,
                   std::vector<ScopeTaskRecord> &scopeMarkers) {
    return AclSpecOptimizer().Collect(model, manager, directory, requests, scopeMarkers);
  }
  aclError UpdateTasks(aclmdlRI model, const std::vector<std::unique_ptr<KernelSpecRequest>> &requests,
                       const std::vector<ScopeTaskRecord> &markersToDisable = {}) {
    return AclSpecOptimizer().UpdateTasks(model, requests, markersToDisable);
  }
};

namespace {
std::vector<aclmdlRITask> MarkerTasks(const std::vector<ScopeTaskRecord> &markers) {
  std::vector<aclmdlRITask> tasks;
  for (const auto &marker : markers) {
    tasks.push_back(marker.task);
  }
  return tasks;
}

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

std::vector<std::string> commitOperations;
std::vector<aclmdlRITask> disabled;
std::vector<aclError> disableResults;

aclError RecordSetTaskParams(aclmdlRITask task, aclmdlRITaskParams *params) {
  commitOperations.emplace_back("SetParams");
  return SetTaskParams(task, params);
}

aclError DisableTask(aclmdlRITask task) {
  commitOperations.emplace_back("Disable");
  const size_t index = disabled.size();
  disabled.push_back(task);
  return index < disableResults.size() ? disableResults[index] : ACL_SUCCESS;
}

aclError RecordUpdateModel(aclmdlRI model) {
  commitOperations.emplace_back("Update");
  return UpdateModel(model);
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
    MOCKER(aclmdlRITaskSetParams).stubs().will(invoke(RecordSetTaskParams));
    MOCKER(aclmdlRITaskDisable).stubs().will(invoke(DisableTask));
    MOCKER(aclmdlRIUpdate).stubs().will(invoke(RecordUpdateModel));
    ut_log::LogBuffer::Instance().Clear();
    commitOperations.clear();
    disabled.clear();
    disableResults.clear();
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
    ut_log::LogBuffer::Instance().Clear();
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
  std::function<void()> onParamCount;
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
size_t binaryQueries = 0;
size_t paramCountQueries = 0;
size_t nameQueries = 0;
size_t copyQueries = 0;
size_t parameterQueries = 0;
size_t lastParameterIndex = 0;
std::vector<aclError> specializationResults;
std::vector<std::string> specializationPaths;
std::vector<std::string> specializationEntries;
size_t loadCalls = 0;
size_t functionCalls = 0;
size_t unloadCalls = 0;
aclError loadResult = ACL_SUCCESS;
aclError functionResult = ACL_SUCCESS;
std::vector<aclError> binaryLoadResults;
std::vector<aclError> functionLookupResults;
std::vector<aclrtBinHandle> binaryLoadHandles;
std::vector<aclrtBinHandle> unloadedBinaries;
bool loadErrorHasHandle = false;
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
  if (binaryLoadHandles.empty()) {
    EXPECT_EQ(binary, &staticBinaryToken);
  } else {
    EXPECT_NE(std::find(binaryLoadHandles.begin(), binaryLoadHandles.end(), binary), binaryLoadHandles.end());
  }
  unloadedBinaries.push_back(binary);
  ++unloadCalls;
}

class AclSpecOptimizerCollectionTest : public testing::Test {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
    MOCKER(aclmdlRITaskSetParams).stubs().will(invoke(RecordSetTaskParams));
    MOCKER(aclmdlRIUpdate).stubs().will(invoke(RecordUpdateModel));
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
    binaryQueries = paramCountQueries = nameQueries = copyQueries = 0;
    identityParamsQueries = identityTypeQueries = 0;
    lastParameterIndex = 0;
    specializationResults.clear();
    specializationPaths.clear();
    specializationEntries.clear();
    loadCalls = functionCalls = 0;
    unloadCalls = 0;
    loadResult = functionResult = ACL_SUCCESS;
    binaryLoadResults.clear();
    functionLookupResults.clear();
    binaryLoadHandles.clear();
    unloadedBinaries.clear();
    loadErrorHasHandle = false;
    setResults.clear();
    updateResults.clear();
    touched.clear();
    commitOperations.clear();
    disabled.clear();
    disableResults.clear();
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
  std::array<uint8_t, 128> args{};
  KernelState kernel;
  CollectorTask task{};
  StreamState stream;
  ModelState model;
  FeatureManager manager;
  AclSpecOptimizerTestAccess collector;
  std::vector<std::unique_ptr<KernelSpecRequest>> requests;
  std::vector<ScopeTaskRecord> scopeMarkers;
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
  specializationEntries.emplace_back(request->kernelEntry == nullptr ? "" : request->kernelEntry);
  return index < specializationResults.size() ? specializationResults[index] : ACL_SUCCESS;
}

extern "C" aclError aclrtBinaryLoadFromFile(const char *, aclrtBinaryLoadOptions *, aclrtBinHandle *binary) {
  const size_t index = loadCalls++;
  const aclError result = index < binaryLoadResults.size() ? binaryLoadResults[index] : loadResult;
  const auto handle = index < binaryLoadHandles.size() ? binaryLoadHandles[index] : &staticBinaryToken;
  *binary = result == ACL_SUCCESS || loadErrorHasHandle ? handle : nullptr;
  return result;
}

namespace {
aclError GetBinaryFunction(aclrtBinHandle, const char *, aclrtFuncHandle *function) {
  const size_t index = functionCalls++;
  *function = &staticFunctionToken;
  return index < functionLookupResults.size() ? functionLookupResults[index] : functionResult;
}

aclError GetStreams(aclmdlRI model, aclrtStream *streams, uint32_t *count) {
  ++streamQueries;
  if (streamQueries == failStreamQuery) {
    return ACL_ERROR_FAILURE;
  }
  const auto &values = static_cast<ModelState *>(model)->streams;
  if (streams == nullptr) {
    *count = values.size();
    return ACL_SUCCESS;
  }
  const size_t available = values.size();
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
    return ACL_ERROR_FAILURE;
  }
  const auto &values = static_cast<StreamState *>(stream)->tasks;
  if (tasks == nullptr) {
    *count = values.size();
    return ACL_SUCCESS;
  }
  const size_t available = values.size();
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
  const auto &value = *static_cast<const KernelState *>(function);
  if (value.onParamCount) {
    value.onParamCount();
  }
  *count = value.paramCount;
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

TEST_F(AclSpecOptimizerCollectionTest, BothModesWarnAndSkipUnreadableTasks) {
  const std::array<const char *, 8> reasons{"null task",
                                            "task type query failed",
                                            "task parameters query failed",
                                            "parameter type mismatch",
                                            "null function handle",
                                            "function name query failed",
                                            "empty function name",
                                            "unterminated function name"};
  KernelState beginKernel, endKernel;
  auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_2201);
  auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_2201);
  for (bool byFlag : {false, true}) {
    ASSERT_TRUE(manager.Init(nullptr, byFlag));
    for (size_t failure = 0; failure < reasons.size(); ++failure) {
      SCOPED_TRACE(reasons[failure]);
      KernelState invalidKernel = kernel;
      CollectorTask invalid = task;
      invalid.params.kernelTaskParams.funcHandle = &invalidKernel;
      aclmdlRITask handle = &invalid;
      switch (failure) {
        case 0:
          handle = nullptr;
          break;
        case 1:
          invalid.typeError = ACL_ERROR_FAILURE;
          break;
        case 2:
          invalid.paramsError = ACL_ERROR_FAILURE;
          break;
        case 3:
          invalid.params.type = ACL_MODEL_RI_TASK_DEFAULT;
          break;
        case 4:
          invalid.params.kernelTaskParams.funcHandle = nullptr;
          break;
        case 5:
          invalidKernel.nameError = ACL_ERROR_FAILURE;
          break;
        case 6:
          invalidKernel.name.clear();
          break;
        case 7:
          invalidKernel.unterminatedName = true;
          break;
      }
      stream.tasks = {&begin, handle, &task, &end};
      ut_log::LogBuffer::Instance().Clear();
      ASSERT_EQ(Collect(), ACL_SUCCESS);
      ASSERT_EQ(requests.size(), 1U);
      EXPECT_EQ(requests.front()->task, &task);
      const auto logs = ut_log::LogBuffer::Instance().GetContent();
      EXPECT_NE(logs.find("[WARN] [SK][ReadTaskIdentity] Task identity unavailable: stream=0 task=1 reason=" +
                          std::string(reasons[failure]) + " ret=" + std::to_string(ACL_ERROR_FAILURE)),
                std::string::npos);
      EXPECT_NE(logs.find("[WARN] [SK][Collect] ReadTaskIdentity failed; skipping task: stream=0 task=1 ret=" +
                          std::to_string(ACL_ERROR_FAILURE)),
                std::string::npos);
    }
  }
}

TEST_F(AclSpecOptimizerCollectionTest, SelectsRangeUnionAndValidatesPairsOnlyForByFlag) {
  struct RangeCase {
    std::string layout;
    std::vector<uint32_t> selected;
    bool valid = true;
  };
  // B/E: markers, K: kernel, O: non-kernel. Equal kernel names still represent distinct tasks.
  const std::vector<RangeCase> cases{{"KK", {}},
                                     {"BE", {}},
                                     {"BEBE", {}},
                                     {"KBKEK", {2}},
                                     {"BKEKBKE", {1, 5}},
                                     {"BBKEKE", {2, 4}},
                                     {"BOKOE", {2}},
                                     {"BOE", {}},
                                     {"BKE", {1}},
                                     {"BEBKE", {3}},
                                     {std::string(64, 'B') + "K" + std::string(64, 'E'), {64}},
                                     {"E", {}, false},
                                     {"BK", {}, false},
                                     {"BBKE", {}, false},
                                     {"BKEE", {}, false}};
  for (bool byFlag : {false, true}) {
    ASSERT_TRUE(manager.Init(nullptr, byFlag));
    for (const auto &entry : cases) {
      SCOPED_TRACE(entry.layout + " byFlag=" + std::to_string(byFlag));
      KernelState beginKernel, endKernel;
      const auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_3510);
      const auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_3510);
      std::vector<CollectorTask> states(entry.layout.size(), task);
      std::vector<aclmdlRITask> expected, markers;
      stream.tasks.clear();
      for (size_t index = 0; index < states.size(); ++index) {
        const char kind = entry.layout[index];
        if (kind == 'B' || kind == 'E') {
          states[index] = kind == 'B' ? begin : end;
          markers.push_back(&states[index]);
        } else if (kind == 'O') {
          states[index].type = ACL_MODEL_RI_TASK_EVENT_WAIT;
          states[index].paramsError = ACL_ERROR_FAILURE;
        } else if (!byFlag || std::find(entry.selected.begin(), entry.selected.end(), index) != entry.selected.end()) {
          expected.push_back(&states[index]);
        }
        stream.tasks.push_back(&states[index]);
      }
      requests.clear();
      requests.push_back(std::make_unique<KernelSpecRequest>());
      auto *previous = requests.front().get();
      scopeMarkers = {{&task}};
      binaryQueries = copyQueries = 0;
      const aclError result = Collect();
      if (byFlag && !entry.valid) {
        EXPECT_EQ(result, ACL_ERROR_INVALID_PARAM);
        ASSERT_EQ(requests.size(), 1U);
        EXPECT_EQ(requests.front().get(), previous);
        EXPECT_EQ(MarkerTasks(scopeMarkers), (std::vector<aclmdlRITask>{&task}));
        EXPECT_EQ(binaryQueries, 0U);
      } else {
        ASSERT_EQ(result, ACL_SUCCESS);
        ASSERT_EQ(requests.size(), expected.size());
        for (size_t index = 0; index < expected.size(); ++index) {
          EXPECT_EQ(requests[index]->task, expected[index]);
        }
        EXPECT_EQ(MarkerTasks(scopeMarkers), markers);
        EXPECT_EQ(binaryQueries, expected.size());
        EXPECT_EQ(copyQueries, expected.size());
      }
    }
  }
}

namespace {
class AclSpecOptimizerScopeTest : public AclSpecOptimizerCollectionTest, public testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    AclSpecOptimizerCollectionTest::SetUp();
    begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_2201);
    end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_2201);
    MOCKER(aclmdlRITaskDisable).stubs().will(invoke(DisableTask));
  }
  aclError RunEntry() {
    return GetParam() ? aclmdlRISpecOptimizeByFlag(&model, nullptr) : aclmdlRISpecOptimize(&model, nullptr);
  }
  KernelState beginKernel, endKernel;
  CollectorTask begin{}, end{};
};
}  // namespace

TEST_P(AclSpecOptimizerScopeTest, MultiStreamSelectionPreservesOtherTasksAndCommitsOnce) {
  CollectorTask before = task, after = task, secondTask = task, unmarkedTask = task, other = task;
  KernelState similarKernel = kernel;
  similarKernel.name = "sk_scope_kernel_begin_dav_2201";
  CollectorTask similar = task;
  similar.params.kernelTaskParams.funcHandle = &similarKernel;
  other.type = ACL_MODEL_RI_TASK_EVENT_RECORD;
  other.paramsError = ACL_ERROR_FAILURE;
  auto secondBegin = begin, secondEnd = end;
  stream.tasks = {&before, &begin, &task, &similar, &other, &end, &after};
  StreamState second{{&secondBegin, &secondTask, &secondEnd}}, empty{{&unmarkedTask}};
  model.streams = {&stream, &empty, &second};
  const auto original = task.params;
  const auto originalArgs = args;
  ut_log::LogBuffer::Instance().Clear();
  ASSERT_EQ(RunEntry(), ACL_SUCCESS);
  std::vector<aclmdlRITask> expected{&task, &similar, &secondTask};
  if (!GetParam()) {
    expected.insert(expected.begin(), &before);
    expected.insert(expected.end() - 1, &after);
    expected.insert(expected.end() - 1, &unmarkedTask);
  }
  EXPECT_EQ(touched, expected);
  EXPECT_EQ(disabled, (std::vector<aclmdlRITask>{&begin, &end, &secondBegin, &secondEnd}));
  EXPECT_EQ(binaryQueries, expected.size());
  EXPECT_EQ(copyQueries, expected.size());
  EXPECT_EQ(updateCount, 1U);
  auto restored = task.params;
  EXPECT_EQ(restored.kernelTaskParams.funcHandle, &staticFunctionToken);
  restored.kernelTaskParams.funcHandle = original.kernelTaskParams.funcHandle;
  EXPECT_EQ(std::memcmp(&restored, &original, sizeof(original)), 0);
  EXPECT_EQ(args, originalArgs);
  if (GetParam()) {
    EXPECT_EQ(std::memcmp(&before.params, &original, sizeof(original)), 0);
    EXPECT_EQ(std::memcmp(&after.params, &original, sizeof(original)), 0);
    const auto logs = ut_log::LogBuffer::Instance().GetContent();
    for (const auto &message :
         {"stream=0, task=2, kernel=AddCustom", "stream=0, task=3, kernel=sk_scope_kernel_begin_dav_2201",
          "stream=2, task=1, kernel=AddCustom"}) {
      const std::string line = "JIT Scope: kernel in scope, " + std::string(message);
      const size_t position = logs.find(line);
      ASSERT_NE(position, std::string::npos);
      EXPECT_EQ(logs.find(line, position + line.size()), std::string::npos);
    }
  }
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagWithoutScopeLeavesOrdinaryTasksUntouched) {
  CollectorTask second = task;
  stream.tasks = {&task, &second};

  ASSERT_EQ(aclmdlRISpecOptimizeByFlag(&model, nullptr), ACL_SUCCESS);
  EXPECT_TRUE(touched.empty());
  EXPECT_TRUE(specializationPaths.empty());
  EXPECT_EQ(loadCalls, 0U);
  EXPECT_EQ(updateCount, 0U);
  EXPECT_TRUE(commitOperations.empty());
}

TEST_F(AclSpecOptimizerCollectionTest, ByFlagRejectsCrossStreamAndLaterInvalidPairingBeforeSubmission) {
  KernelState beginKernel, endKernel;
  auto begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_2201);
  auto end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_2201);
  CollectorTask first = task, second = task;

  const auto check = [&](const std::vector<std::vector<aclmdlRITask>> &layouts) {
    model.streams.clear();
    touched.clear();
    specializationPaths.clear();
    commitOperations.clear();
    disabled.clear();
    requests.clear();
    scopeMarkers.clear();
    loadCalls = functionCalls = updateCount = 0;
    std::vector<StreamState> streams;
    streams.reserve(layouts.size());
    for (const auto &layout : layouts) {
      streams.push_back({layout});
    }
    for (auto &streamState : streams) {
      model.streams.push_back(&streamState);
    }

    EXPECT_EQ(aclmdlRISpecOptimizeByFlag(&model, nullptr), ACL_ERROR_INVALID_PARAM);
    EXPECT_TRUE(touched.empty());
    EXPECT_TRUE(specializationPaths.empty());
    EXPECT_EQ(loadCalls, 0U);
    EXPECT_EQ(updateCount, 0U);
    EXPECT_TRUE(commitOperations.empty());
  };

  check(std::vector<std::vector<aclmdlRITask>>{{&begin, &first}, {&end, &second}});
  check(std::vector<std::vector<aclmdlRITask>>{{&begin, &first, &end}, {&end, &second}});
}

TEST_P(AclSpecOptimizerScopeTest, InvalidInputsAndInitializationErrorsStopBeforeTaskQueries) {
  aclmdlRISpecOptions options{};
  options.specCompileOptionCount = 1;
  const auto entry = GetParam() ? aclmdlRISpecOptimizeByFlag : aclmdlRISpecOptimize;
  EXPECT_EQ(entry(&model, &options), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(entry(nullptr, nullptr), ACL_ERROR_INVALID_PARAM);
  SkUtSetAclmdlRIGetIdRet(ACL_ERROR_BAD_ALLOC);
  EXPECT_EQ(entry(&model, nullptr), ACL_ERROR_BAD_ALLOC);
  EXPECT_EQ(streamQueries, 0U);
}

TEST_P(AclSpecOptimizerScopeTest, ConvertsExceptionsWithoutCompilingOrMutatingModel) {
  stream.tasks = {&begin, &task, &end};
  const std::array<aclError, 5> expected{ACL_ERROR_BAD_ALLOC, ACL_ERROR_BAD_ALLOC, ACL_ERROR_INVALID_PARAM,
                                         ACL_ERROR_FAILURE, ACL_ERROR_FAILURE};
  for (size_t kind = 0; kind < expected.size(); ++kind) {
    kernel.onParamCount = [kind]() {
      switch (kind) {
        case 0:
          throw std::bad_alloc();
        case 1:
          throw std::length_error("test");
        case 2:
          throw std::invalid_argument("test");
        case 3:
          throw std::runtime_error("test");
        default:
          throw 1;
      }
    };
    EXPECT_EQ(RunEntry(), expected[kind]);
  }
  EXPECT_TRUE(specializationPaths.empty());
  EXPECT_EQ(loadCalls, 0U);
  EXPECT_TRUE(commitOperations.empty());
}

INSTANTIATE_TEST_SUITE_P(PublicEntries, AclSpecOptimizerScopeTest, testing::Bool());

namespace {
class AclSpecOptimizerSubmissionTest : public AclSpecOptimizerCollectionTest {
 protected:
  void SetUp() override {
    AclSpecOptimizerCollectionTest::SetUp();
    begin = MakeMarker(beginKernel, SPEC_SCOPE_BEGIN_DAV_2201);
    end = MakeMarker(endKernel, SPEC_SCOPE_END_DAV_2201);
    secondKernel = kernel;
    second = task;
    second.params.kernelTaskParams.funcHandle = &secondKernel;
    stream.tasks = {&begin, &task, &second, &end};
    binaryLoadHandles = {&tokens[0], &tokens[1]};
    MOCKER(aclmdlRITaskDisable).stubs().will(invoke(DisableTask));
  }

  aclError RunEntry(bool byFlag) {
    return byFlag ? aclmdlRISpecOptimizeByFlag(&model, nullptr) : aclmdlRISpecOptimize(&model, nullptr);
  }

  void ExpectSubmission(size_t rewrites, size_t disables, size_t updates, size_t released,
                        bool callbackFailed = false) {
    std::vector<std::string> operations(rewrites, "SetParams");
    operations.insert(operations.end(), disables, "Disable");
    operations.insert(operations.end(), updates, "Update");
    EXPECT_EQ(commitOperations, operations);
    EXPECT_EQ(unloadedBinaries.size(), released);
    const bool referenced = rewrites != 0;
    EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), referenced && !callbackFailed ? 1U : 0U);
    if (referenced && !callbackFailed) {
      ASSERT_EQ(SkUtInvokeModelDestroyCallback(&model), ACL_SUCCESS);
      EXPECT_EQ(unloadedBinaries.size(), loadCalls);
    }
    if (!referenced) {
      EXPECT_EQ(task.params.kernelTaskParams.funcHandle, &kernel);
      EXPECT_EQ(second.params.kernelTaskParams.funcHandle, &secondKernel);
    }
    EXPECT_EQ(begin.params.kernelTaskParams.funcHandle, &beginKernel);
    EXPECT_EQ(end.params.kernelTaskParams.funcHandle, &endKernel);
    for (const auto &path : specializationPaths) {
      EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(path).parent_path()));
    }
  }

  KernelState beginKernel, endKernel, secondKernel;
  CollectorTask begin{}, end{}, second{};
  std::array<int, 2> tokens{};
};

struct CandidateCase {
  const char *name;
  size_t taskCount, unsupportedCount;
  std::vector<aclError> compileResults;
  size_t rewrites;
  bool hasMarkers = true;
};
class AclSpecOptimizerCandidateTest : public AclSpecOptimizerSubmissionTest,
                                      public testing::WithParamInterface<std::tuple<bool, CandidateCase>> {};

struct LoadFailureCase {
  const char *name;
  std::vector<aclError> loadResults, lookupResults;
  bool errorHasHandle;
  size_t released;
};
class AclSpecOptimizerLoadFailureTest : public AclSpecOptimizerSubmissionTest,
                                        public testing::WithParamInterface<std::tuple<bool, LoadFailureCase>> {};

struct CommitFailureCase {
  const char *name;
  std::vector<aclError> setResults, disableResults, updateResults;
  aclError callbackResult;
  size_t rewrites, disables, updates;
};
class AclSpecOptimizerCommitFailureTest : public AclSpecOptimizerSubmissionTest,
                                          public testing::WithParamInterface<std::tuple<bool, CommitFailureCase>> {};
}  // namespace

TEST_P(AclSpecOptimizerCandidateTest, KeepsSkippedTasksAndCleansMarkersAfterCompilation) {
  const auto &expected = std::get<1>(GetParam());
  stream.tasks.clear();
  if (expected.hasMarkers) {
    stream.tasks.push_back(&begin);
  }
  const std::array<aclmdlRITask, 2> tasks{&task, &second};
  ASSERT_LE(expected.taskCount, tasks.size());
  ASSERT_LE(expected.unsupportedCount, expected.taskCount);
  stream.tasks.insert(stream.tasks.end(), tasks.begin(), tasks.begin() + expected.taskCount);
  if (expected.hasMarkers) {
    stream.tasks.push_back(&end);
  }
  if (expected.unsupportedCount > 0) {
    kernel.binary.clear();
  }
  if (expected.unsupportedCount > 1) {
    secondKernel.binary.clear();
  }
  specializationResults = expected.compileResults;

  EXPECT_EQ(RunEntry(std::get<0>(GetParam())), ACL_SUCCESS);
  const size_t markerCount = expected.hasMarkers ? 2U : 0U;
  ExpectSubmission(expected.rewrites, markerCount, expected.hasMarkers ? 1U : 0U, 0);
  EXPECT_EQ(specializationPaths.size(), expected.taskCount - expected.unsupportedCount);
  EXPECT_EQ(loadCalls, expected.rewrites);
  EXPECT_EQ(functionCalls, expected.rewrites);
  const std::vector<aclmdlRITask> expectedMarkers =
      expected.hasMarkers ? std::vector<aclmdlRITask>{&begin, &end} : std::vector<aclmdlRITask>{};
  EXPECT_EQ(disabled, expectedMarkers);
  if (expected.unsupportedCount > 0) {
    EXPECT_EQ(task.params.kernelTaskParams.funcHandle, &kernel);
  }
}

INSTANTIATE_TEST_SUITE_P(
    PublicEntries, AclSpecOptimizerCandidateTest,
    testing::Combine(testing::Bool(),
                     testing::Values(
                         // name, task count, unsupported count, compile results, rewrites, has markers
                         CandidateCase{"Success", 2, 0, {}, 2}, CandidateCase{"MarkersOnly", 0, 0, {}, 0},
                         CandidateCase{"Empty", 0, 0, {}, 0, false}, CandidateCase{"MixedSupport", 2, 1, {}, 1},
                         CandidateCase{"AllUnsupported", 2, 2, {}, 0},
                         CandidateCase{"PartialCompileFailure", 2, 0, {ACL_SUCCESS, ACL_ERROR_FAILURE}, 1},
                         CandidateCase{"AllCompileFailure", 2, 0, {ACL_ERROR_FAILURE, ACL_ERROR_FAILURE}, 0})),
    [](const testing::TestParamInfo<AclSpecOptimizerCandidateTest::ParamType> &info) {
      return std::string(std::get<0>(info.param) ? "ByFlag" : "Optimize") + std::get<1>(info.param).name;
    });

TEST_P(AclSpecOptimizerLoadFailureTest, DoesNotMutateTasksAndReleasesLoadedBinaries) {
  const auto &expected = std::get<1>(GetParam());
  binaryLoadResults = expected.loadResults;
  functionLookupResults = expected.lookupResults;
  loadErrorHasHandle = expected.errorHasHandle;

  EXPECT_EQ(RunEntry(std::get<0>(GetParam())), ACL_ERROR_INVALID_PARAM);
  ExpectSubmission(0, 0, 0, expected.released);
}

INSTANTIATE_TEST_SUITE_P(
    PublicEntries, AclSpecOptimizerLoadFailureTest,
    testing::Combine(testing::Bool(),
                     testing::Values(
                         // name, load results, lookup results, error has handle, released binaries
                         LoadFailureCase{"FirstLoad", {ACL_ERROR_INVALID_PARAM}, {}, false, 0},
                         LoadFailureCase{"LaterLoad", {ACL_SUCCESS, ACL_ERROR_INVALID_PARAM}, {}, false, 1},
                         LoadFailureCase{"FunctionLookup", {}, {ACL_SUCCESS, ACL_ERROR_INVALID_PARAM}, false, 2},
                         LoadFailureCase{"LoadErrorWithHandle", {ACL_ERROR_INVALID_PARAM}, {}, true, 1})),
    [](const testing::TestParamInfo<AclSpecOptimizerLoadFailureTest::ParamType> &info) {
      return std::string(std::get<0>(info.param) ? "ByFlag" : "Optimize") + std::get<1>(info.param).name;
    });

TEST_P(AclSpecOptimizerCommitFailureTest, StopsAtFailureAndRetainsReferencedBinaries) {
  const auto &expected = std::get<1>(GetParam());
  setResults = expected.setResults;
  disableResults = expected.disableResults;
  updateResults = expected.updateResults;
  SkUtSetAclmdlRIDestroyRegisterCallbackRet(expected.callbackResult);

  EXPECT_EQ(RunEntry(std::get<0>(GetParam())), ACL_ERROR_INVALID_PARAM);
  ExpectSubmission(expected.rewrites, expected.disables, expected.updates, 0, expected.callbackResult != ACL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    PublicEntries, AclSpecOptimizerCommitFailureTest,
    testing::Combine(
        testing::Bool(),
        testing::Values(
            // name, SetParams/Disable/Update results, callback result, attempted rewrites/disables/updates
            CommitFailureCase{"FirstSet", {ACL_ERROR_INVALID_PARAM}, {}, {}, ACL_SUCCESS, 1, 0, 0},
            CommitFailureCase{"LaterSet", {ACL_SUCCESS, ACL_ERROR_INVALID_PARAM}, {}, {}, ACL_SUCCESS, 2, 0, 0},
            CommitFailureCase{"FirstDisable", {}, {ACL_ERROR_INVALID_PARAM}, {}, ACL_SUCCESS, 2, 1, 0},
            CommitFailureCase{"LaterDisable", {}, {ACL_SUCCESS, ACL_ERROR_INVALID_PARAM}, {}, ACL_SUCCESS, 2, 2, 0},
            CommitFailureCase{"Update", {}, {}, {ACL_ERROR_INVALID_PARAM}, ACL_SUCCESS, 2, 2, 1},
            CommitFailureCase{"Callback", {}, {}, {}, ACL_ERROR_INVALID_PARAM, 2, 2, 1},
            CommitFailureCase{"DisableAndCallback", {}, {ACL_ERROR_INVALID_PARAM}, {}, ACL_ERROR_FAILURE, 2, 1, 0},
            CommitFailureCase{"UpdateAndCallback", {}, {}, {ACL_ERROR_INVALID_PARAM}, ACL_ERROR_FAILURE, 2, 2, 1})),
    [](const testing::TestParamInfo<AclSpecOptimizerCommitFailureTest::ParamType> &info) {
      return std::string(std::get<0>(info.param) ? "ByFlag" : "Optimize") + std::get<1>(info.param).name;
    });

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

}  // namespace model_spec
