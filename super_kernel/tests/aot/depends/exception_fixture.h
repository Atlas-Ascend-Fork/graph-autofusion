/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <vector>

#include "acl/acl.h"
#include "dump/adump_pub.h"

namespace sk::test {
// An external Runtime exception. Args are copied verbatim from the real RI task;
// the fixture never decodes or constructs SuperKernel's private argument layout.
class Exception {
 public:
  explicit Exception(aclmdlRITask task);
  ~Exception();
  Exception(const Exception &) = delete;
  Exception &operator=(const Exception &) = delete;
  void Raise(bool nullInfo = false);
  uint32_t Dump(Adx::ExceptionDumpInfo *output, uint32_t capacity, uint32_t *count, Adx::ExceptionDumpMode *mode,
                bool nullInfo = false);
  aclrtExceptionInfo info{};
  aclrtFuncHandle function = nullptr;
  void *args = nullptr;
  uint32_t argsSize = 0;
  aclError functionResult = ACL_SUCCESS;
  aclError argsResult = ACL_SUCCESS;
  rtError_t registersResult = 0;
  std::vector<rtExceptionErrRegInfo_t> registers;

 private:
  void *storage_ = nullptr;
};
void RegisterExceptionCallback(aclrtExceptionInfoCallbackFunc callback);
Exception *FindException(const void *info);
// Register a standard ELF symbol table before Optimize consumes this binary.
uint64_t RegisterExceptionBinary(aclmdlRITask task);
bool ExceptionBinaryBuffer(void *binary, void **buffer, uint32_t *size);
}  // namespace sk::test
