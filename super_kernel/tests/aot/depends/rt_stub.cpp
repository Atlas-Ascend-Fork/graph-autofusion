/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file rt_stub.cpp
 * @brief Stub implementations for runtime functions used in unit tests
 *        Only contains runtime-specific functions (rt*), not ACL functions (aclrt*)
 *        For UT stub mode, all functions return success (0) by default
 */

#include "acl/acl.h"
#include "ut_common_stubs.h"
#include "model_fixture_internal.h"
#include "exception_fixture.h"
#include "runtime/kernel.h"
#include "runtime/base.h"
#include <cstring>
#include <cstdio>
extern "C" {

// Error codes
#define RT_SUCCESS 0
#define RT_ERROR -1

// rtBinaryGetMetaNum - stub implementation for unit tests
// Always returns success for UT stub mode
int rtBinaryGetMetaNum(void *binHdl, int type_enum, size_t *metaNum) {
  if (metaNum != nullptr) {
    *metaNum = type_enum == RT_BINARY_TYPE_SK_INFO ? sk::test::BinaryMetadataCount(binHdl) : 0;
  }
  return RT_SUCCESS;
}

// rtBinaryGetMetaInfo - stub implementation for unit tests
// Always returns success for UT stub mode
int rtBinaryGetMetaInfo(void *binHdl, int type_enum, size_t metaNum, void **data_list, size_t *size_list) {
  int result;
  if (type_enum == RT_BINARY_TYPE_SK_INFO && sk::test::BinaryMetadata(binHdl, metaNum, data_list, size_list, result)) {
    return result;
  }
  (void)binHdl;
  (void)type_enum;
  (void)metaNum;
  (void)data_list;
  (void)size_list;
  return RT_SUCCESS;
}

// rtGetBinBuffer - stub implementation for unit tests
int rtGetBinBuffer(void *binHdl, int addrType, void **buffer, uint32_t *size) {
  if (addrType == RT_BIN_HOST_ADDR && buffer != nullptr && size != nullptr &&
      sk::test::ExceptionBinaryBuffer(binHdl, buffer, size)) {
    return RT_SUCCESS;
  }
  if (buffer != nullptr) {
    *buffer = nullptr;
  }
  if (size != nullptr) {
    *size = 0;
  }
  return RT_SUCCESS;
}

// rtBinaryGetMetaData - stub implementation for unit tests
// Always returns success for UT stub mode
int rtBinaryGetMetaData(void *binHdl, int type_enum, size_t metaNum, void **data_list, size_t *size_list) {
  (void)binHdl;
  (void)type_enum;
  (void)metaNum;
  (void)data_list;
  (void)size_list;
  return RT_SUCCESS;
}

// Super kernel capture/replay API stubs
void rt_sk_start_capture(void) {
  // Stub implementation
}

void rt_sk_stop_capture(void) {
  // Stub implementation
}

void rt_sk_capture_snapshot(void) {
  // Stub implementation
}

void rt_sk_replay(void) {
  // Stub implementation
}

// rtGetExceptionRegInfo - stub implementation for unit tests
// Returns success with no cores in error
rtError_t rtGetExceptionRegInfo(const void *exception, rtExceptionErrRegInfo_t **errRegInfo, uint32_t *coreNum) {
  if (auto *fixture = sk::test::FindException(exception)) {
    *errRegInfo = fixture->registers.data();
    *coreNum = static_cast<uint32_t>(fixture->registers.size());
    return fixture->registersResult;
  }
  (void)exception;
  (void)errRegInfo;
  *coreNum = 0;
  return RT_SUCCESS;
}

static uint32_t g_simtAivType = 0;
static uint32_t g_functionAllocUbufSize = 0;
static int g_rtFunctionGetMetaInfoRet = RT_SUCCESS;

void SetSimtAivType(uint32_t aivType) {
  g_simtAivType = aivType;
}

void SetFunctionAllocUbufSize(uint32_t allocUbufSize) {
  g_functionAllocUbufSize = allocUbufSize;
}

void SetRtFunctionGetMetaInfoRet(int ret) {
  g_rtFunctionGetMetaInfoRet = ret;
}

rtError_t rtFunctionGetMetaInfo(void *funcHandle, int type_enum, void *data, uint32_t length) {
  (void)funcHandle;
  (void)length;
  if (type_enum == RT_FUNCTION_TYPE_AIV_TYPE_FLAG && data != nullptr) {
    *reinterpret_cast<uint32_t *>(data) = g_simtAivType;
    return RT_SUCCESS;
  }
  if (type_enum == RT_FUNCTION_TYPE_COMPILER_ALLOC_UB_SIZE && data != nullptr) {
    if (g_rtFunctionGetMetaInfoRet != RT_SUCCESS) {
      return g_rtFunctionGetMetaInfoRet;
    }
    *reinterpret_cast<uint32_t *>(data) = g_functionAllocUbufSize;
    return RT_SUCCESS;
  }
  if (data != nullptr) {
    *reinterpret_cast<uint32_t *>(data) = 0;
  }
  return RT_SUCCESS;
}

aclError aclmdlRIDebugJsonPrint(aclmdlRI model, const char *path, uint32_t flag) {
  (void)model;
  (void)flag;
  SkUtRecordDebugJsonPrintPath(path);
  return ACL_SUCCESS;
}

}  // extern "C"
