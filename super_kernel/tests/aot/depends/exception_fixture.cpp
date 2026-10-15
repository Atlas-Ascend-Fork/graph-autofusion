/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "exception_fixture.h"

#include <elf.h>

#include <array>
#include <cstring>
#include <map>
#include <stdexcept>

#include "runtime/kernel.h"

namespace {
aclrtExceptionInfoCallbackFunc exceptionCallback = nullptr;
Adx::ExceptionDumpCallback dumpCallback = nullptr;
thread_local sk::test::Exception *activeException = nullptr;
struct BinarySymbols {
  Elf64_Ehdr header{};
  std::array<Elf64_Shdr, 4> sections{};
  std::array<Elf64_Sym, 9> symbols{};
  char sectionNames[32] = "\0.shstrtab\0.strtab\0.symtab";
  char symbolNames[32] = "\0fixture_subkernel";
};
std::map<void *, BinarySymbols> exceptionBinaries;
class ActiveException {
 public:
  explicit ActiveException(sk::test::Exception *value) : previous_(activeException) {
    activeException = value;
  }
  ~ActiveException() {
    activeException = previous_;
  }

 private:
  sk::test::Exception *previous_;
};
}  // namespace
namespace Adx {
int32_t AdumpRegExceptionDumpCallback(ExceptionDumpCallback callback) {
  dumpCallback = callback;
  return ACL_SUCCESS;
}
}  // namespace Adx
namespace sk::test {
uint64_t RegisterExceptionBinary(aclmdlRITask task) {
  aclmdlRITaskParams params{};
  aclrtBinHandle binary = nullptr;
  void *device = nullptr;
  size_t deviceSize = 0;
  size_t count = 0;
  if (aclmdlRITaskGetParams(task, &params) != ACL_SUCCESS ||
      aclrtFunctionGetBinary(params.kernelTaskParams.funcHandle, &binary) != ACL_SUCCESS || binary == nullptr ||
      aclrtBinaryGetDevAddress(binary, &device, &deviceSize) != ACL_SUCCESS ||
      rtBinaryGetMetaNum(binary, RT_BINARY_TYPE_SK_INFO, &count) != 0 || count == 0 || count > 2) {
    throw std::runtime_error("Cannot read Runtime binary metadata");
  }
  // Runtime SK metadata transport from model_fixture.cpp: reserved u32,
  // capability u64, original entry u64, then four SK entry u64s. Consume
  // the external payload rather than reconstructing production DFX data.
  std::array<std::array<unsigned char, 52>, 2> metadata{};
  void *buffers[] = {metadata[0].data(), metadata[1].data()};
  size_t sizes[] = {metadata[0].size(), metadata[1].size()};
  if (rtBinaryGetMetaInfo(binary, RT_BINARY_TYPE_SK_INFO, count, buffers, sizes) != 0) {
    throw std::runtime_error("Cannot read Runtime SK entries");
  }
  auto &elf = exceptionBinaries[binary];
  std::memcpy(elf.header.e_ident, ELFMAG, SELFMAG);
  elf.header.e_ident[EI_CLASS] = ELFCLASS64;
  elf.header.e_ident[EI_DATA] = ELFDATA2LSB;
  elf.header.e_ident[EI_VERSION] = EV_CURRENT;
  elf.header.e_type = ET_REL;
  elf.header.e_version = EV_CURRENT;
  elf.header.e_ehsize = sizeof(Elf64_Ehdr);
  elf.header.e_shoff = offsetof(BinarySymbols, sections);
  elf.header.e_shentsize = sizeof(Elf64_Shdr);
  elf.header.e_shnum = elf.sections.size();
  elf.header.e_shstrndx = 1;
  auto &names = elf.sections[1];
  names.sh_name = 1;
  names.sh_type = SHT_STRTAB;
  names.sh_offset = offsetof(BinarySymbols, sectionNames);
  names.sh_size = sizeof(elf.sectionNames);
  auto &strings = elf.sections[2];
  strings.sh_name = 11;
  strings.sh_type = SHT_STRTAB;
  strings.sh_offset = offsetof(BinarySymbols, symbolNames);
  strings.sh_size = sizeof(elf.symbolNames);
  auto &symbols = elf.sections[3];
  symbols.sh_name = 19;
  symbols.sh_type = SHT_SYMTAB;
  symbols.sh_offset = offsetof(BinarySymbols, symbols);
  symbols.sh_size = sizeof(Elf64_Sym) * (1 + count * 4);
  symbols.sh_entsize = sizeof(Elf64_Sym);
  symbols.sh_link = 2;
  symbols.sh_info = 1;
  symbols.sh_addralign = alignof(Elf64_Sym);
  for (size_t core = 0; core < count; ++core) {
    for (size_t entry = 0; entry < 4; ++entry) {
      auto &symbol = elf.symbols[1 + core * 4 + entry];
      std::memcpy(&symbol.st_value, metadata[core].data() + 20 + entry * sizeof(uint64_t), sizeof(uint64_t));
      symbol.st_name = 1;
      symbol.st_info = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC);
      symbol.st_shndx = SHN_ABS;
      symbol.st_size = 8;
    }
  }
  return reinterpret_cast<uintptr_t>(device) + elf.symbols[1].st_value;
}
bool ExceptionBinaryBuffer(void *binary, void **buffer, uint32_t *size) {
  auto found = exceptionBinaries.find(binary);
  if (found == exceptionBinaries.end()) {
    return false;
  }
  *buffer = &found->second;
  *size = sizeof(found->second);
  return true;
}
Exception::Exception(aclmdlRITask task) {
  aclmdlRITaskParams params{};
  if (aclmdlRITaskGetParams(task, &params) != ACL_SUCCESS) {
    throw std::runtime_error("Cannot read exception task");
  }
  function = params.kernelTaskParams.funcHandle;
  argsSize = params.kernelTaskParams.argsSize;
  info.expandInfo.type = RT_EXCEPTION_AICORE;
  if (argsSize != 0) {
    if (aclrtMalloc(&storage_, argsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      throw std::runtime_error("Cannot allocate exception arguments");
    }
    if (aclrtMemcpy(storage_, argsSize, params.kernelTaskParams.args, argsSize,
                    params.kernelTaskParams.isHostArgs ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_DEVICE) !=
        ACL_SUCCESS) {
      aclrtFree(storage_);
      throw std::runtime_error("Cannot copy exception arguments");
    }
    args = storage_;
  }
}
Exception::~Exception() {
  if (storage_ != nullptr) {
    aclrtFree(storage_);
  }
}
void Exception::Raise(bool nullInfo) {
  if (exceptionCallback == nullptr) {
    throw std::runtime_error("Production exception callback was not registered");
  }
  ActiveException active(this);
  exceptionCallback(nullInfo ? nullptr : &info);
}
uint32_t Exception::Dump(Adx::ExceptionDumpInfo *output, uint32_t capacity, uint32_t *count,
                         Adx::ExceptionDumpMode *mode, bool nullInfo) {
  if (dumpCallback == nullptr) {
    throw std::runtime_error("Production dump callback was not registered");
  }
  ActiveException active(this);
  return dumpCallback(nullInfo ? nullptr : &info, output, capacity, count, mode);
}
void RegisterExceptionCallback(aclrtExceptionInfoCallbackFunc callback) {
  exceptionCallback = callback;
}
Exception *FindException(const void *info) {
  return activeException != nullptr && info == &activeException->info ? activeException : nullptr;
}
}  // namespace sk::test
