/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <cstring>
#include <limits>
#include <thread>
#include <unistd.h>
#include <vector>
#include <elf.h>
#include "mockcpp/mockcpp.hpp"
#include "acl/acl.h"
#include "sk_common.h"
#include "ut_common_stubs.h"

#define private public
#define protected public
/*!
 * \file test_sk_common.cpp
 * \brief Unit tests for sk_common.cpp host helpers (GetSocName, GetCurrentSkKernelArch).
 */

namespace {

class SkCommonTest : public testing::Test {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
  }
  void TearDown() override {
    SkUtResetTestControls();
  }
};

std::vector<uint8_t> BuildMinimalValidElf64() {
  std::vector<uint8_t> buffer(sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3 + 128, 0);

  Elf64_Ehdr *ehdr = reinterpret_cast<Elf64_Ehdr *>(buffer.data());
  ehdr->e_ident[EI_MAG0] = ELFMAG0;
  ehdr->e_ident[EI_MAG1] = ELFMAG1;
  ehdr->e_ident[EI_MAG2] = ELFMAG2;
  ehdr->e_ident[EI_MAG3] = ELFMAG3;
  ehdr->e_ident[EI_CLASS] = ELFCLASS64;
  ehdr->e_shoff = sizeof(Elf64_Ehdr);
  ehdr->e_shnum = 3;
  ehdr->e_shstrndx = 2;

  Elf64_Shdr *shdr = reinterpret_cast<Elf64_Shdr *>(buffer.data() + ehdr->e_shoff);

  shdr[0].sh_name = 0;
  shdr[0].sh_type = SHT_NULL;

  shdr[1].sh_name = 1;
  shdr[1].sh_type = SHT_SYMTAB;
  shdr[1].sh_offset = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3;
  shdr[1].sh_size = sizeof(Elf64_Sym);

  shdr[2].sh_name = 10;
  shdr[2].sh_type = SHT_STRTAB;
  shdr[2].sh_offset = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3 + sizeof(Elf64_Sym);
  shdr[2].sh_size = 20;

  char *shstrtab = reinterpret_cast<char *>(buffer.data() + shdr[2].sh_offset);
  shstrtab[1] = '.';
  shstrtab[2] = 's';
  shstrtab[3] = 'y';
  shstrtab[4] = 'm';
  shstrtab[5] = 't';
  shstrtab[6] = 'a';
  shstrtab[7] = 'b';
  shstrtab[10] = '.';
  shstrtab[11] = 's';
  shstrtab[12] = 't';
  shstrtab[13] = 'r';
  shstrtab[14] = 't';
  shstrtab[15] = 'a';
  shstrtab[16] = 'b';

  Elf64_Sym *symtab = reinterpret_cast<Elf64_Sym *>(buffer.data() + shdr[1].sh_offset);
  symtab[0].st_name = 0;
  symtab[0].st_value = 0;
  symtab[0].st_size = 0;
  symtab[0].st_info = 0;

  char *strtab = reinterpret_cast<char *>(buffer.data() + shdr[2].sh_offset + shdr[2].sh_size);

  return buffer;
}

// Fake aclrtGetSocName implementations for mockcpp stubs.
const char *FakeSocName_Null() {
  return nullptr;
}
const char *FakeSocName_Ascend910B() {
  return "Ascend910B";
}
const char *FakeSocName_Ascend950() {
  return "Ascend950PG";
}
const char *FakeSocName_Ascend950Exact() {
  return "Ascend950";
}
const char *FakeSocName_Ascend910() {
  return "Ascend910";
}
const char *FakeSocName_Empty() {
  return "";
}

void ExitIsolatedSkRuntimeConfigTest() {
  GlobalMockObject::verify();
  fflush(nullptr);
  _exit(::testing::Test::HasFailure() ? 1 : 0);
}

}  // namespace

TEST_F(SkCommonTest, SkKernelArchToString_ReturnsExpectedName) {
  EXPECT_STREQ(to_string(SkKernelArch::DAV_2201), "DAV_2201");
  EXPECT_STREQ(to_string(SkKernelArch::DAV_3510), "DAV_3510");
  EXPECT_STREQ(to_string(SkKernelArch::UNKNOWN), "UNKNOWN");
}

TEST_F(SkCommonTest, GetSkKernelArchSymbolSuffix_ReturnsExpectedSuffix) {
  EXPECT_STREQ(GetSkKernelArchSymbolSuffix(SkKernelArch::DAV_2201), "dav_2201");
  EXPECT_STREQ(GetSkKernelArchSymbolSuffix(SkKernelArch::DAV_3510), "dav_3510");
  EXPECT_STREQ(GetSkKernelArchSymbolSuffix(SkKernelArch::UNKNOWN), "unknown");
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_NullBinAddr_ReturnsFalse) {
  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;

  bool ret =
      GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000), nullptr, 100, 0x10, symbolName, funcSize, symbolBind);
  EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_ZeroBinSize_ReturnsFalse) {
  char buffer[100] = {0};
  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;

  bool ret =
      GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000), buffer, 0, 0x10, symbolName, funcSize, symbolBind);
  EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_BinSizeSmallerThanElfHeader_ReturnsFalse) {
  char buffer[10] = {0};
  buffer[0] = ELFMAG0;
  buffer[1] = ELFMAG1;
  buffer[2] = ELFMAG2;
  buffer[3] = ELFMAG3;

  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;

  bool ret =
      GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000), buffer, 10, 0x10, symbolName, funcSize, symbolBind);
  EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_SectionHeaderExceedsBinSize_ReturnsFalse) {
  size_t bufferSize = sizeof(Elf64_Ehdr);
  std::vector<uint8_t> buffer(bufferSize, 0);

  Elf64_Ehdr *ehdr = reinterpret_cast<Elf64_Ehdr *>(buffer.data());
  ehdr->e_ident[EI_MAG0] = ELFMAG0;
  ehdr->e_ident[EI_MAG1] = ELFMAG1;
  ehdr->e_ident[EI_MAG2] = ELFMAG2;
  ehdr->e_ident[EI_MAG3] = ELFMAG3;
  ehdr->e_ident[EI_CLASS] = ELFCLASS64;
  ehdr->e_shoff = 1000;
  ehdr->e_shnum = 10;
  ehdr->e_shstrndx = 0;

  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;

  bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000), reinterpret_cast<char *>(buffer.data()),
                               buffer.size(), 0x10, symbolName, funcSize, symbolBind);
  EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_SectionHeaderTableSizeExceedsBinSize_ReturnsFalse) {
  size_t bufferSize = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr);
  std::vector<uint8_t> buffer(bufferSize, 0);

  Elf64_Ehdr *ehdr = reinterpret_cast<Elf64_Ehdr *>(buffer.data());
  ehdr->e_ident[EI_MAG0] = ELFMAG0;
  ehdr->e_ident[EI_MAG1] = ELFMAG1;
  ehdr->e_ident[EI_MAG2] = ELFMAG2;
  ehdr->e_ident[EI_MAG3] = ELFMAG3;
  ehdr->e_ident[EI_CLASS] = ELFCLASS64;
  ehdr->e_shoff = sizeof(Elf64_Ehdr);
  ehdr->e_shnum = 100;
  ehdr->e_shstrndx = 0;

  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;

  bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000), reinterpret_cast<char *>(buffer.data()),
                               buffer.size(), 0x10, symbolName, funcSize, symbolBind);
  EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_InvalidShstrndx_ReturnsFalse) {
  size_t bufferSize = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 2;
  std::vector<uint8_t> buffer(bufferSize, 0);

  Elf64_Ehdr *ehdr = reinterpret_cast<Elf64_Ehdr *>(buffer.data());
  ehdr->e_ident[EI_MAG0] = ELFMAG0;
  ehdr->e_ident[EI_MAG1] = ELFMAG1;
  ehdr->e_ident[EI_MAG2] = ELFMAG2;
  ehdr->e_ident[EI_MAG3] = ELFMAG3;
  ehdr->e_ident[EI_CLASS] = ELFCLASS64;
  ehdr->e_shoff = sizeof(Elf64_Ehdr);
  ehdr->e_shnum = 2;
  ehdr->e_shstrndx = 10;

  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;

  bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000), reinterpret_cast<char *>(buffer.data()),
                               buffer.size(), 0x10, symbolName, funcSize, symbolBind);
  EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_ShstrtabExceedsBinSize_ReturnsFalse) {
  size_t bufferSize = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3;
  std::vector<uint8_t> buffer(bufferSize, 0);

  Elf64_Ehdr *ehdr = reinterpret_cast<Elf64_Ehdr *>(buffer.data());
  ehdr->e_ident[EI_MAG0] = ELFMAG0;
  ehdr->e_ident[EI_MAG1] = ELFMAG1;
  ehdr->e_ident[EI_MAG2] = ELFMAG2;
  ehdr->e_ident[EI_MAG3] = ELFMAG3;
  ehdr->e_ident[EI_CLASS] = ELFCLASS64;
  ehdr->e_shoff = sizeof(Elf64_Ehdr);
  ehdr->e_shnum = 3;
  ehdr->e_shstrndx = 2;

  Elf64_Shdr *shdr = reinterpret_cast<Elf64_Shdr *>(buffer.data() + ehdr->e_shoff);
  shdr[2].sh_offset = 10000;
  shdr[2].sh_size = 100;

  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;

  bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000), reinterpret_cast<char *>(buffer.data()),
                               buffer.size(), 0x10, symbolName, funcSize, symbolBind);
  EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_SectionRangeOverflow_ReturnsFalse) {
  constexpr size_t bufferSize = 512;
  constexpr size_t shstrtabOffset = 320;
  constexpr size_t symtabOffset = 384;
  constexpr uint64_t strtabOffset = 500;
  std::vector<uint8_t> buffer(bufferSize, 0);

  Elf64_Ehdr *ehdr = reinterpret_cast<Elf64_Ehdr *>(buffer.data());
  ehdr->e_ident[EI_MAG0] = ELFMAG0;
  ehdr->e_ident[EI_MAG1] = ELFMAG1;
  ehdr->e_ident[EI_MAG2] = ELFMAG2;
  ehdr->e_ident[EI_MAG3] = ELFMAG3;
  ehdr->e_ident[EI_CLASS] = ELFCLASS64;
  ehdr->e_shoff = sizeof(Elf64_Ehdr);
  ehdr->e_shnum = 4;
  ehdr->e_shstrndx = 3;

  Elf64_Shdr *shdr = reinterpret_cast<Elf64_Shdr *>(buffer.data() + ehdr->e_shoff);
  shdr[0].sh_type = SHT_NULL;
  shdr[1].sh_name = 1;
  shdr[1].sh_type = SHT_SYMTAB;
  shdr[1].sh_offset = symtabOffset;
  shdr[1].sh_size = sizeof(Elf64_Sym);
  shdr[2].sh_name = 9;
  shdr[2].sh_type = SHT_STRTAB;
  shdr[2].sh_offset = strtabOffset;
  shdr[2].sh_size = std::numeric_limits<uint64_t>::max() - strtabOffset + 1;
  shdr[3].sh_name = 17;
  shdr[3].sh_type = SHT_STRTAB;
  shdr[3].sh_offset = shstrtabOffset;
  shdr[3].sh_size = 27;

  char *shstrtab = reinterpret_cast<char *>(buffer.data() + shstrtabOffset);
  std::memcpy(shstrtab + 1, ".symtab", sizeof(".symtab"));
  std::memcpy(shstrtab + 9, ".strtab", sizeof(".strtab"));
  std::memcpy(shstrtab + 17, ".shstrtab", sizeof(".shstrtab"));

  Elf64_Sym *symtab = reinterpret_cast<Elf64_Sym *>(buffer.data() + symtabOffset);
  symtab[0].st_name = 0;
  symtab[0].st_value = 0x10;
  symtab[0].st_size = 8;
  symtab[0].st_info = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC);
  std::memcpy(buffer.data() + strtabOffset, "func", sizeof("func"));

  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;
  bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x3001), reinterpret_cast<char *>(buffer.data()),
                               buffer.size(), 0x10, symbolName, funcSize, symbolBind);

  EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_MissingSymtabAndStrtab_ReturnsFalse) {
  size_t bufferSize = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3 + 50;
  std::vector<uint8_t> buffer(bufferSize, 0);

  Elf64_Ehdr *ehdr = reinterpret_cast<Elf64_Ehdr *>(buffer.data());
  ehdr->e_ident[EI_MAG0] = ELFMAG0;
  ehdr->e_ident[EI_MAG1] = ELFMAG1;
  ehdr->e_ident[EI_MAG2] = ELFMAG2;
  ehdr->e_ident[EI_MAG3] = ELFMAG3;
  ehdr->e_ident[EI_CLASS] = ELFCLASS64;
  ehdr->e_shoff = sizeof(Elf64_Ehdr);
  ehdr->e_shnum = 3;
  ehdr->e_shstrndx = 2;

  Elf64_Shdr *shdr = reinterpret_cast<Elf64_Shdr *>(buffer.data() + ehdr->e_shoff);
  shdr[0].sh_type = SHT_NULL;
  shdr[1].sh_type = SHT_PROGBITS;
  shdr[2].sh_type = SHT_STRTAB;
  shdr[2].sh_offset = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3;
  shdr[2].sh_size = 50;

  char *shstrtab = reinterpret_cast<char *>(buffer.data() + shdr[2].sh_offset);
  shstrtab[1] = '.';
  shstrtab[2] = 't';
  shstrtab[3] = 'e';
  shstrtab[4] = 'x';
  shstrtab[5] = 't';

  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;

  bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000), reinterpret_cast<char *>(buffer.data()),
                               buffer.size(), 0x10, symbolName, funcSize, symbolBind);
  EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_DifferentBinHandleBuildsSeparateCache) {
  auto buffer = BuildMinimalValidElf64();

  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;

  bool ret1 = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1001), reinterpret_cast<char *>(buffer.data()),
                                buffer.size(), 0x10, symbolName, funcSize, symbolBind);

  bool ret2 = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1002), reinterpret_cast<char *>(buffer.data()),
                                buffer.size(), 0x10, symbolName, funcSize, symbolBind);

  EXPECT_FALSE(ret1);
  EXPECT_FALSE(ret2);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_SameBinHandleUsesCache) {
  auto buffer = BuildMinimalValidElf64();

  std::string symbolName1, symbolName2;
  uint64_t funcSize1 = 0, funcSize2 = 0;
  std::string symbolBind1, symbolBind2;

  aclrtBinHandle binHdl = reinterpret_cast<aclrtBinHandle>(0x2001);

  bool ret1 = GetFuncSymbolInfo(binHdl, reinterpret_cast<char *>(buffer.data()), buffer.size(), 0x10, symbolName1,
                                funcSize1, symbolBind1);

  bool ret2 = GetFuncSymbolInfo(binHdl, reinterpret_cast<char *>(buffer.data()), buffer.size(), 0x10, symbolName2,
                                funcSize2, symbolBind2);

  EXPECT_EQ(ret1, ret2);
  EXPECT_EQ(symbolName1, symbolName2);
  EXPECT_EQ(funcSize1, funcSize2);
  EXPECT_EQ(symbolBind1, symbolBind2);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_ConcurrentCacheAccessReturnsConsistentResults) {
  constexpr uint32_t threadCount = 16;
  constexpr uint32_t handlesPerThread = 64;
  auto buffer = BuildMinimalValidElf64();
  std::atomic<uint32_t> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> resultsConsistent{true};
  std::vector<std::thread> threads;

  for (uint32_t threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
    threads.emplace_back([&, threadIndex]() {
      ready.fetch_add(1, std::memory_order_relaxed);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (uint32_t i = 0; i < handlesPerThread; ++i) {
        const uintptr_t handleValue = 0x100000U + threadIndex * handlesPerThread + i;
        std::string symbolName;
        uint64_t funcSize = 0;
        std::string symbolBind;
        bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(handleValue),
                                     reinterpret_cast<const char *>(buffer.data()), buffer.size(), 0x10, symbolName,
                                     funcSize, symbolBind);
        if (ret || !symbolName.empty() || funcSize != 0 || !symbolBind.empty()) {
          resultsConsistent.store(false, std::memory_order_relaxed);
        }
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != threadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(resultsConsistent.load(std::memory_order_relaxed));
}

TEST_F(SkCommonTest, GetDeviceCoreNums_NoDevices_ReturnsInvalidParam) {
  SkUtSetAclrtGetDeviceRet(ACL_ERROR_INVALID_PARAM);
  int64_t cubeNum = -1, vecNum = -1;
  EXPECT_EQ(GetDeviceCoreNums(cubeNum, vecNum), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(cubeNum, 0);
  EXPECT_EQ(vecNum, -1);
}

TEST_F(SkCommonTest, GetDeviceCoreNums_GetDeviceFails_FallsBackToDeviceZero) {
  SkUtSetAclrtGetDeviceRet(ACL_ERROR_INVALID_PARAM);
  SkUtSetDeviceCount(1);
  int64_t cubeNum = 0, vecNum = 0;
  EXPECT_EQ(GetDeviceCoreNums(cubeNum, vecNum), ACL_SUCCESS);
  EXPECT_EQ(cubeNum, 32);
  EXPECT_EQ(vecNum, 32);
}

TEST_F(SkCommonTest, GetDeviceCoreNums_GetDeviceCountFails_ReturnsInvalidParam) {
  SkUtSetAclrtGetDeviceRet(ACL_ERROR_INVALID_PARAM);
  SkUtSetDeviceCount(1);
  SkUtSetAclrtGetDeviceCountRet(ACL_ERROR_FAILURE);
  int64_t cubeNum = -1, vecNum = -1;
  EXPECT_EQ(GetDeviceCoreNums(cubeNum, vecNum), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(cubeNum, 0);
  EXPECT_EQ(vecNum, -1);
}

TEST_F(SkCommonTest, GetDeviceCoreNums_GetDeviceInfoFails_ReturnsInvalidParam) {
  SkUtSetAclrtGetDeviceInfoRet(ACL_ERROR_INVALID_PARAM);
  int64_t cubeNum = -1, vecNum = -1;
  EXPECT_EQ(GetDeviceCoreNums(cubeNum, vecNum), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(cubeNum, 0);
  EXPECT_EQ(vecNum, -1);
}

TEST_F(SkCommonTest, GetDeviceCoreNums_Success_ReturnsValidValues) {
  int64_t cubeNum = 0, vecNum = 0;
  aclError ret = GetDeviceCoreNums(cubeNum, vecNum);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_GT(cubeNum, 0);
  EXPECT_GT(vecNum, 0);
}

class SkCommonSocTest : public testing::Test {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
  }

  void TearDown() override {
    GlobalMockObject::verify();
    SkUtResetTestControls();
  }
};

// ==================== GetSocName ====================

TEST_F(SkCommonSocTest, GetSocName_ReturnsAscend910BFromDefaultStub) {
  // 默认 stub 返回 "Ascend910B"，无需 mock
  EXPECT_EQ(GetSocName(), "Ascend910B");
}

TEST_F(SkCommonSocTest, GetSocName_NullptrReturnsEmptyString) {
  MOCKER(aclrtGetSocName).stubs().will(invoke(FakeSocName_Null));
  EXPECT_EQ(GetSocName(), "");
}

TEST_F(SkCommonSocTest, GetSocName_EmptyStringIsReturnedAsIs) {
  MOCKER(aclrtGetSocName).stubs().will(invoke(FakeSocName_Empty));
  EXPECT_EQ(GetSocName(), "");
}

TEST_F(SkCommonSocTest, GetSocName_Ascend950Variant) {
  MOCKER(aclrtGetSocName).stubs().will(invoke(FakeSocName_Ascend950));
  EXPECT_EQ(GetSocName(), "Ascend950PG");
}

// ==================== GetCurrentSkKernelArch ====================

TEST_F(SkCommonSocTest, GetCurrentSkKernelArch_DefaultStubIsDav2201) {
  // 默认 stub 返回 "Ascend910B"，不匹配 "Ascend950"
  InitSkRuntimeConfig();
  EXPECT_EQ(GetCurrentSkKernelArch(), SkKernelArch::DAV_2201);
}

TEST_F(SkCommonSocTest, GetCurrentSkKernelArch_NullptrFallsBackToDav2201) {
  MOCKER(aclrtGetSocName).stubs().will(invoke(FakeSocName_Null));
  InitSkRuntimeConfig();
  EXPECT_EQ(GetCurrentSkKernelArch(), SkKernelArch::DAV_2201);
}

TEST_F(SkCommonSocTest, GetCurrentSkKernelArch_EmptyStringIsDav2201) {
  MOCKER(aclrtGetSocName).stubs().will(invoke(FakeSocName_Empty));
  InitSkRuntimeConfig();
  EXPECT_EQ(GetCurrentSkKernelArch(), SkKernelArch::DAV_2201);
}

TEST_F(SkCommonSocTest, GetCurrentSkKernelArch_Ascend910IsDav2201) {
  MOCKER(aclrtGetSocName).stubs().will(invoke(FakeSocName_Ascend910));
  InitSkRuntimeConfig();
  EXPECT_EQ(GetCurrentSkKernelArch(), SkKernelArch::DAV_2201);
}

TEST_F(SkCommonSocTest, GetCurrentSkKernelArch_ExactAscend950IsDav3510) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        MOCKER(aclrtGetSocName).stubs().will(invoke(FakeSocName_Ascend950Exact));
        InitSkRuntimeConfig();
        EXPECT_EQ(GetCurrentSkKernelArch(), SkKernelArch::DAV_3510);
        ExitIsolatedSkRuntimeConfigTest();
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(SkCommonSocTest, GetCurrentSkKernelArch_Ascend950WithSuffixIsDav3510) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        // 后缀变体（如 "Ascend950PG"）也应被识别
        MOCKER(aclrtGetSocName).stubs().will(invoke(FakeSocName_Ascend950));
        InitSkRuntimeConfig();
        EXPECT_EQ(GetCurrentSkKernelArch(), SkKernelArch::DAV_3510);
        ExitIsolatedSkRuntimeConfigTest();
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(SkCommonSocTest, InitSkRuntimeConfig_DefaultStubUsesDav2201Defaults) {
  InitSkRuntimeConfig();
  const SkRuntimeConfig &config = GetSkRuntimeConfig();
  EXPECT_EQ(config.kernelArch, SkKernelArch::DAV_2201);
  EXPECT_EQ(config.eventCoreNum, SK_EVENT_DAV_2201_CORE_NUM);
  EXPECT_EQ(config.tickUsMultiplier, SK_DAV_2201_TICK_US_MULTIPLIER);
}

TEST_F(SkCommonSocTest, InitSkRuntimeConfig_Ascend950UsesDav3510Config) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        MOCKER(aclrtGetSocName).stubs().will(invoke(FakeSocName_Ascend950Exact));
        InitSkRuntimeConfig();
        const SkRuntimeConfig &config = GetSkRuntimeConfig();
        EXPECT_EQ(config.kernelArch, SkKernelArch::DAV_3510);
        EXPECT_EQ(config.eventCoreNum, SK_EVENT_DAV_3510_CORE_NUM);
        EXPECT_EQ(config.tickUsMultiplier, SK_DAV_3510_TICK_US_MULTIPLIER);
        ExitIsolatedSkRuntimeConfigTest();
      },
      ::testing::ExitedWithCode(0), "");
}
