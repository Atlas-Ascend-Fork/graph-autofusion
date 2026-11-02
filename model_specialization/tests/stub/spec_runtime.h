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
 * \file spec_runtime.h
 * \brief Specialization-specific Runtime and ACLRTC declarations supplementing shared test stubs.
 */

#ifndef SPEC_TEST_RUNTIME_H
#define SPEC_TEST_RUNTIME_H

#include <stdint.h>
#include "acl/acl.h"
#include "acl/acl_rt_compile.h"

#undef ACL_SUCCESS
#undef ACL_ERROR_INVALID_PARAM

static const int ACL_ERROR_BAD_ALLOC = 200000;
static const int ACL_ERROR_INTERNAL_ERROR = 500000;
static const int ACL_ERROR_RT_INSUFFICIENT_INPUT_ARRAY = 107039;

static const int ACL_ERROR_FEATURE_UNSUPPORTED = 200006;

extern "C" {
typedef struct aclrtcKernelSpecRequest {
  const char *resourceId;
  const char *kernelEntry;
  uint64_t argsCount;
  const void *const *argsAddr;
  const uint64_t *argsBytes;
  const char *const *options;
  uint64_t optionCount;
  const char *const *skOptions;
  uint64_t skOptionCount;
  uint8_t reserved[256];
} aclrtcKernelSpecRequest;

aclError aclrtcKernelSpecialization(const aclrtcKernelSpecRequest *request, const char *outputPath);
aclError aclrtBinaryLoadFromFile(const char *path, aclrtBinaryLoadOptions *options, aclrtBinHandle *binary);
aclError aclrtFunctionGetParamCount(const void *function, size_t *count);
aclError aclrtFunctionGetParamInfo(const void *function, size_t index, size_t *offset, size_t *size);
}

#endif
