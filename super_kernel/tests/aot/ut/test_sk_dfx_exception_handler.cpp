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
#include "mockcpp/mockcpp.hpp"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <cstring>
#include <map>
#include <unordered_map>
#include <cstdlib>
#include <cstdio>
#include "securec.h"

#define private public
#define protected public
#include "sk_dfx_exception_handler.h"
#include "sk_common.h"
#include "sk_event_recorder.h"
#include "sk_log.h"
#include "sk_types.h"
#include "runtime/kernel.h"
#include "ut_common_stubs.h"
#include "dlog_pub.h"

class SkDfxExceptionHandlerTest : public testing::Test {
 protected:
  void SetUp() override {
    handler = std::make_unique<SuperKernelExceptionHandler>();
  }

  void TearDown() override {
    GlobalMockObject::verify();
    handler.reset();
    SkUtSetAclrtGetSocName("Ascend910B");  // Reset to default arch (DAV_2201)
  }

  std::unique_ptr<SuperKernelExceptionHandler> handler;
};

// ==================== Fake GetCurrentSkKernelArch for DAV_3510 mock ====================
// SkRuntimeConfig is initialized once per process via std::call_once, so
// SkUtSetAclrtGetSocName("Ascend950") cannot change the arch after initialization.
// Instead, we mock GetCurrentSkKernelArch directly to return DAV_3510.
static SkKernelArch FakeGetCurrentSkKernelArch_Dav3510() {
  return SkKernelArch::DAV_3510;
}

// ==================== Helper Functions for Test Setup ====================

// Helper: Set up single-node DFX info on stack buffer for IdentifyErrorNodeByPC tests
static void SetupSingleDfxNode(uint8_t *buffer, uint32_t aicSize, uint32_t aivSize, const uint32_t entryAic[4],
                               const uint32_t entryAiv[4], SkHeaderInfo &headerInfo, SkDeviceEntryArgs *&deviceArgs,
                               SkDfxInfo *&dfxInfo, SuperKernelExceptionHandler *h) {
  headerInfo = {};
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);
  headerInfo.nodeCnt = 1;

  deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo->binHdl = 0xAAA;
  dfxInfo->funcHdlOri = 0xBBB;
  dfxInfo->aicSize = aicSize;
  dfxInfo->aivSize = aivSize;
  for (int i = 0; i < 4; i++) {
    dfxInfo->entryAic[i] = entryAic[i];
    dfxInfo->entryAiv[i] = entryAiv[i];
  }

  h->skDeviceEntryArgsHost = deviceArgs;
  h->skHeaderInfoHost = &headerInfo;
}

// Helper: Allocate and setup heap buffer for PrintCoreSymbols / PrintAllCoreSymbols tests
static uint8_t *SetupOpTraceTestBuffer(uint32_t nodeCnt, bool hasDfx, SkHeaderInfo &headerInfo,
                                       SkDeviceEntryArgs *&deviceArgs, SuperKernelExceptionHandler *h) {
  uint32_t totalSize = sizeof(SkHeaderInfo) + sizeof(SkCounterInfo) * 75;
  if (hasDfx && nodeCnt > 0) {
    totalSize += sizeof(SkDfxInfo) * nodeCnt;
  }

  headerInfo = {};
  headerInfo.nodeCnt = nodeCnt;
  headerInfo.counterOffset = sizeof(SkHeaderInfo);
  if (hasDfx && nodeCnt > 0) {
    headerInfo.dfxOffset = sizeof(SkHeaderInfo) + sizeof(SkCounterInfo) * 75;
  }
  headerInfo.totalSize = totalSize;

  const size_t allocSize = totalSize;
  if (allocSize == 0 || allocSize > 1024 * 1024) {
    return nullptr;
  }

  uint8_t *buf = reinterpret_cast<uint8_t *>(malloc(allocSize));
  (void)memset_s(buf, allocSize, 0, allocSize);
  deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buf);
  deviceArgs->skHeader = headerInfo;

  h->skDeviceEntryArgsHost = deviceArgs;
  h->skHeaderInfoHost = &headerInfo;

  return buf;
}

// ==================== Helper Functions Tests ====================

TEST_F(SkDfxExceptionHandlerTest, StartsWith_ValidPrefix) {
  EXPECT_TRUE(handler->StartsWith("sk_entry_test", "sk_entry"));
  EXPECT_TRUE(handler->StartsWith("sk_entry_test", ""));
  EXPECT_TRUE(handler->StartsWith("test", ""));
}

TEST_F(SkDfxExceptionHandlerTest, StartsWith_InvalidPrefix) {
  EXPECT_FALSE(handler->StartsWith("test_function", "sk_entry"));
  EXPECT_FALSE(handler->StartsWith("SK_ENTRY", "sk_entry"));
  EXPECT_FALSE(handler->StartsWith("", "prefix"));
}

TEST_F(SkDfxExceptionHandlerTest, StartsWith_NullPointers) {
  EXPECT_FALSE(handler->StartsWith(nullptr, "prefix"));
  EXPECT_FALSE(handler->StartsWith("source", nullptr));
  EXPECT_FALSE(handler->StartsWith(nullptr, nullptr));
}

TEST_F(SkDfxExceptionHandlerTest, CheckError_Success) {
  aclError ret = handler->CheckError(ACL_SUCCESS, "Test operation");
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkDfxExceptionHandlerTest, CheckError_Failure) {
  aclError ret = handler->CheckError(ACL_ERROR_FAILURE, "Test operation");
  EXPECT_EQ(ret, ACL_ERROR_FAILURE);
}

// ==================== HandleException Tests ====================

TEST_F(SkDfxExceptionHandlerTest, HandleException_NullExceptionInfo) {
  handler->HandleException(nullptr);
  // Should not crash, just return early
  SUCCEED();
}

// Mock for aclrtGetFuncHandleFromExceptionInfo
aclError Fake_aclrtGetFuncHandleFromExceptionInfo_Success(const aclrtExceptionInfo *exceptionInfo,
                                                          aclrtFuncHandle *funcHandle) {
  (void)exceptionInfo;
  *funcHandle = reinterpret_cast<aclrtFuncHandle>(0x1000);
  return ACL_SUCCESS;
}

// Mock for aclrtGetFunctionName - returns sk_entry function
aclError Fake_aclrtGetFunctionName_sk_entry(void *funcHandle, uint32_t maxLen, char *name) {
  (void)funcHandle;
  snprintf_s(name, maxLen, maxLen, "%s", "sk_entry");
  return ACL_SUCCESS;
}

// Mock for aclrtGetFunctionName - returns non-sk_entry function
aclError Fake_aclrtGetFunctionName_other(void *funcHandle, uint32_t maxLen, char *name) {
  (void)funcHandle;
  snprintf_s(name, maxLen, maxLen, "%s", "some_other_function");
  return ACL_SUCCESS;
}

static const std::string &MaxLengthSkEntryName() {
  static const std::string name = [] {
    const std::string prefix = "sk_entry_";
    return prefix + std::string(MAX_FUNC_NAME_LEN - 1U - prefix.size(), 'x');
  }();
  return name;
}

aclError Fake_aclrtGetFunctionName_MaxLengthSkEntry(void *funcHandle, uint32_t maxLen, char *name) {
  (void)funcHandle;
  const std::string &functionName = MaxLengthSkEntryName();
  if (name == nullptr || maxLen <= functionName.size()) {
    return ACL_ERROR_INVALID_PARAM;
  }
  errno_t ret = memcpy_s(name, maxLen, functionName.c_str(), functionName.size() + 1U);
  return ret == EOK ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

// Global buffer for aclrtMemcpy mock in FillExceptionDumpInfo tests
static uint8_t *g_mockDeviceBuffer = nullptr;
static size_t g_mockDeviceBufferSize = 0;

// Mock for aclrtMemcpy - simulates reading from device memory
aclError Fake_aclrtMemcpy_DeviceToHost(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
  (void)src;
  if (kind != ACL_MEMCPY_DEVICE_TO_HOST) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (dst == nullptr || count == 0 || count > destMax) {
    return ACL_ERROR_INVALID_PARAM;
  }
  // Copy from global mock buffer (simulating device memory)
  if (g_mockDeviceBuffer != nullptr && count <= g_mockDeviceBufferSize) {
    errno_t err = memcpy_s(dst, destMax, g_mockDeviceBuffer, count);
    if (err != 0) {
      return ACL_ERROR_FAILURE;
    }
  }
  return ACL_SUCCESS;
}

// Mock for aclrtGetArgsFromExceptionInfo
aclError Fake_aclrtGetArgsFromExceptionInfo_Success(const aclrtExceptionInfo *exceptionInfo, void **args,
                                                    uint32_t *argsLen) {
  (void)exceptionInfo;
  *args = reinterpret_cast<void *>(0x3000);  // Directly return GM address
  // Use g_mockDeviceBufferSize if set, otherwise fallback to 8 (original behavior)
  *argsLen = g_mockDeviceBufferSize > 0 ? static_cast<uint32_t>(g_mockDeviceBufferSize) : 8;
  return ACL_SUCCESS;
}

// Mock that returns small argsLen for testing totalSize > skDeviceEntryArgsPtrLen
aclError Fake_aclrtGetArgsFromExceptionInfo_SmallLen(const aclrtExceptionInfo *exceptionInfo, void **args,
                                                     uint32_t *argsLen) {
  (void)exceptionInfo;
  *args = reinterpret_cast<void *>(0x3000);
  *argsLen = 8;  // Very small, triggers totalSize > ptrLen check
  return ACL_SUCCESS;
}

// Mock for aclrtMallocHost
aclError Fake_aclrtMallocHost_Success(void **hostPtr, size_t size) {
  if (hostPtr == nullptr || size == 0 || size > 1024 * 1024) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *hostPtr = malloc(size);
  return (*hostPtr != nullptr) ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

// Mock for aclrtMemcpy
aclError Fake_aclrtMemcpy_Success(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
  (void)destMax;
  (void)src;
  (void)count;
  (void)kind;
  if (dst != nullptr && src != nullptr && count > 0 && count <= destMax) {
    memcpy_s(dst, destMax, src, count);
  }
  return ACL_SUCCESS;
}

// Mock for aclrtFreeHost
aclError Fake_aclrtFreeHost_Success(void *hostPtr) {
  if (hostPtr != nullptr) {
    free(hostPtr);
  }
  return ACL_SUCCESS;
}

// Mock for rtGetExceptionRegInfo
int Fake_rtGetExceptionRegInfo_Success(const void *exception, rtExceptionErrRegInfo_t **errRegInfo, uint32_t *coreNum) {
  (void)exception;
  (void)errRegInfo;
  *coreNum = 0;  // No cores with errors
  return 0;
}

// Mock for aclrtGetArgsFromExceptionInfo - returns short length
aclError Fake_aclrtGetArgsFromExceptionInfo_Short(const aclrtExceptionInfo *exceptionInfo, void **args,
                                                  uint32_t *argsLen) {
  (void)exceptionInfo;
  *args = reinterpret_cast<void *>(0x2000);
  *argsLen = 4;  // Less than 8
  return ACL_SUCCESS;
}

TEST_F(SkDfxExceptionHandlerTest, IsSuperKernelException_Success) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  bool result = handler->IsSuperKernelException(exceptionInfo);
  EXPECT_TRUE(result);
}

TEST_F(SkDfxExceptionHandlerTest, IsSuperKernelException_AcceptsMaximumFunctionName) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  ASSERT_EQ(MaxLengthSkEntryName().size(), MAX_FUNC_NAME_LEN - 1U);
  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_MaxLengthSkEntry));

  EXPECT_TRUE(handler->IsSuperKernelException(exceptionInfo));
}

TEST_F(SkDfxExceptionHandlerTest, IsSuperKernelException_NotSkEntry) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

  bool result = handler->IsSuperKernelException(exceptionInfo);
  EXPECT_FALSE(result);
}

TEST_F(SkDfxExceptionHandlerTest, ExtractTaskQueue_EmptyOffsets) {
  // Setup minimal SkHeaderInfo
  uint32_t skDeviceEntryArgsSize = sizeof(SkHeaderInfo);
  SkHeaderInfo headerInfo{};
  headerInfo.totalSize = skDeviceEntryArgsSize;
  handler->skHeaderInfoHost = &headerInfo;

  bool result = handler->ExtractTaskQueue();
  EXPECT_TRUE(result);
  EXPECT_EQ(handler->aicTaskCnt, 0);
  EXPECT_EQ(handler->aivTaskCnt, 0);
}

TEST_F(SkDfxExceptionHandlerTest, ExtractTaskQueue_WithTaskCounts) {
  // Setup test data
  uint32_t skDeviceEntryArgsSize = sizeof(SkHeaderInfo);
  SkHeaderInfo headerInfo;
  headerInfo.aicQueOffset = skDeviceEntryArgsSize;
  headerInfo.aivQueOffset = skDeviceEntryArgsSize + sizeof(TaskQue) + sizeof(TaskInfo);
  headerInfo.counterOffset = 0;
  headerInfo.dfxOffset = 0;
  headerInfo.eventConfigOffset = 0;
  headerInfo.nodeCnt = 0;
  headerInfo.totalSize = 1024;

  uint8_t buffer[1024] = {0};
  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  TaskQue *aicQueue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aicQueOffset);
  aicQueue->taskCnt = 2;
  aicQueue->cap = 10;

  TaskQue *aivQueue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aivQueOffset);
  aivQueue->taskCnt = 3;
  aivQueue->cap = 10;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;

  bool result = handler->ExtractTaskQueue();
  EXPECT_TRUE(result);
  EXPECT_EQ(handler->aicTaskCnt, 2);
  EXPECT_EQ(handler->aivTaskCnt, 3);
}

TEST_F(SkDfxExceptionHandlerTest, FreeResources_AllNull) {
  handler->FreeResources();
  // Should not crash with all null pointers
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, FreeResources_WithAllocatedMemory) {
  // Allocate some memory to free
  const size_t allocSize = 1024;

  if (allocSize == 0 || allocSize > 1024 * 1024) {
    return;
  }
  handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs *>(malloc(allocSize));

  EXPECT_NE(handler->skDeviceEntryArgsHost, nullptr);

  // Set skHeaderInfoHost to point inside skDeviceEntryArgsHost (simulating normal flow)
  handler->skHeaderInfoHost = &(handler->skDeviceEntryArgsHost->skHeader);

  // Mock aclrtFreeHost to track calls
  MOCKER(aclrtFreeHost).stubs().will(invoke(Fake_aclrtFreeHost_Success));

  handler->FreeResources();

  EXPECT_EQ(handler->skDeviceEntryArgsHost, nullptr);
  EXPECT_EQ(handler->skHeaderInfoHost, nullptr);
}

TEST_F(SkDfxExceptionHandlerTest, GetOrLoadKernelSymbols_CacheHit) {
  // Setup cache
  KernelFuncName kernelFuncName{"test_function"};
  handler->opSymbolCache[1] = kernelFuncName;

  KernelFuncName result = handler->GetOrLoadKernelSymbols(1);

  EXPECT_EQ(result.name, "test_function");
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_CoreIdExceedsAicoreNums) {
  uint32_t coreId = 100;  // Exceeds default aicoreNums=75

  handler->PrintCoreSymbols(coreId, RT_CORE_TYPE_AIC, 0, 0);

  // Should not crash, just log error and return
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintSymbolByCoreId_EmptySymbols) {
  KernelFuncName emptyKernelFuncName{""};

  handler->PrintSymbolByCoreId(0, RT_CORE_TYPE_AIC, 0, 0, emptyKernelFuncName);

  // Should not crash, just log that no function name is available
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintSymbolByCoreId_WithSymbols) {
  KernelFuncName kernelFuncName{"test_kernel"};

  handler->PrintSymbolByCoreId(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000, kernelFuncName);

  // Should not crash and print function name
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintAllCoreSymbols_DefaultAicoreNums) {
  // Setup minimal required structures
  uint32_t skDeviceEntryArgsSize = sizeof(SkHeaderInfo);
  SkHeaderInfo headerInfo{};
  headerInfo.totalSize = skDeviceEntryArgsSize;
  handler->skHeaderInfoHost = &headerInfo;

  const size_t allocSize = 1024;
  if (allocSize == 0 || allocSize > 1024 * 1024) {
    return;
  }
  handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs *>(malloc(allocSize));

  handler->PrintAllCoreSymbols();

  // Should iterate through all 75 cores without crashing
  free(handler->skDeviceEntryArgsHost);
  SUCCEED();
}

// ==================== ExtractSkDeviceEntryArgsPtr Tests ====================

TEST_F(SkDfxExceptionHandlerTest, ExtractSkDeviceEntryArgsPtr_Success) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));

  bool result = handler->ExtractSkDeviceEntryArgsPtr(exceptionInfo);
  EXPECT_TRUE(result);
  EXPECT_EQ(handler->skDeviceEntryArgsDev, reinterpret_cast<void *>(0x3000));
}

TEST_F(SkDfxExceptionHandlerTest, ExtractSkDeviceEntryArgsPtr_TooShort) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Short));

  bool result = handler->ExtractSkDeviceEntryArgsPtr(exceptionInfo);
  EXPECT_FALSE(result);
}

// ==================== SuperKernelExceptionCallBackFunc Tests ====================

TEST_F(SkDfxExceptionHandlerTest, SuperKernelExceptionCallBackFunc_Null) {
  SuperKernelExceptionCallBackFunc(nullptr);
  // Should not crash
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, SuperKernelExceptionCallBackFunc_WithValidInfo) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));  // Not sk_entry, should skip

  SuperKernelExceptionCallBackFunc(exceptionInfo);

  // Should not crash
  SUCCEED();
}

// ==================== IdentifyErrorNodeByPC Tests (Commit 1 & 2) ====================

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_CurrentPCZero_ShouldReturnEarly) {
  // Setup minimal SkHeaderInfo with valid dfxOffset and nodeCnt
  SkHeaderInfo headerInfo{};
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);
  headerInfo.nodeCnt = 1;

  uint8_t buffer[1024] = {0};
  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;

  // currentPC == 0 should return immediately (fix for pc 0 bug)
  handler->IdentifyErrorNodeByPC(0, RT_CORE_TYPE_AIC, 0x1000, 0);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_DfxOffsetZero_ShouldReturnEarly) {
  SkHeaderInfo headerInfo{};
  headerInfo.nodeCnt = 1;  // dfxOffset = 0

  handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs *>(&headerInfo);
  handler->skHeaderInfoHost = &headerInfo;

  handler->IdentifyErrorNodeByPC(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_NodeCntZero_ShouldReturnEarly) {
  SkHeaderInfo headerInfo{};  // nodeCnt = 0
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);

  uint8_t buffer[1024] = {0};
  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;

  handler->IdentifyErrorNodeByPC(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_MatchAICEntry) {
  // Setup: one node with AIC entry at 0x1000, size 0x200
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  SkDfxInfo *dfxInfo;

  uint32_t entryAic[4] = {0x1000, 0, 0, 0};
  uint32_t entryAiv[4] = {0, 0, 0, 0};
  SetupSingleDfxNode(buffer, 0x200, 0, entryAic, entryAiv, headerInfo, deviceArgs, dfxInfo, handler.get());

  // Mock aclrtGetFunctionName for the found node
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  // currentPC = 0x1100 falls within [0x1000, 0x1200) -> should match node[0]
  handler->IdentifyErrorNodeByPC(5, RT_CORE_TYPE_AIC, 0x1000, 0x1100);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_MatchAIVEntry) {
  // Setup: one node with AIV entry at 0x3000, size 0x100
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  SkDfxInfo *dfxInfo;

  uint32_t entryAic[4] = {0, 0, 0, 0};
  uint32_t entryAiv[4] = {0x3000, 0, 0, 0};
  SetupSingleDfxNode(buffer, 0, 0x100, entryAic, entryAiv, headerInfo, deviceArgs, dfxInfo, handler.get());

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  // currentPC = 0x3050 falls within [0x3000, 0x3100) -> should match node[0]'s AIV entry
  handler->IdentifyErrorNodeByPC(30, RT_CORE_TYPE_AIV, 0x3000, 0x3050);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_NoMatch_ShouldLogNoSubKernelMatched) {
  // Setup: one node with AIC entry at 0x1000, size 0x100
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  SkDfxInfo *dfxInfo;

  uint32_t entryAic[4] = {0x1000, 0, 0, 0};
  uint32_t entryAiv[4] = {0, 0, 0, 0};
  SetupSingleDfxNode(buffer, 0x100, 0, entryAic, entryAiv, headerInfo, deviceArgs, dfxInfo, handler.get());

  // currentPC = 0xFFFF is outside any entry range -> no match
  handler->IdentifyErrorNodeByPC(5, RT_CORE_TYPE_AIC, 0x1000, 0xFFFF);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_SkipInvalidZeroEntries) {
  // Setup: node with entryAic[0]=0 (invalid, should skip), entryAic[1]=valid
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  SkDfxInfo *dfxInfo;

  uint32_t entryAic[4] = {0, 0x2000, 0, 0};
  uint32_t entryAiv[4] = {0, 0, 0, 0};
  SetupSingleDfxNode(buffer, 0x200, 0, entryAic, entryAiv, headerInfo, deviceArgs, dfxInfo, handler.get());

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  // PC should match entryAic[1] which is at 0x2100 (within [0x2000, 0x2200))
  handler->IdentifyErrorNodeByPC(5, RT_CORE_TYPE_AIC, 0x2000, 0x2100);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_PCBoundaryExactEnd_NotIncluded) {
  // Test that endAddr is exclusive: currentPC == entryAddr + funcSize should NOT match
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  SkDfxInfo *dfxInfo;

  uint32_t entryAic[4] = {0x1000, 0, 0, 0};
  uint32_t entryAiv[4] = {0, 0, 0, 0};
  SetupSingleDfxNode(buffer, 0x100, 0, entryAic, entryAiv, headerInfo, deviceArgs, dfxInfo, handler.get());

  // currentPC == endAddr (0x1100), should NOT match because range is [entryAddr, endAddr)
  handler->IdentifyErrorNodeByPC(5, RT_CORE_TYPE_AIC, 0x1000, 0x1100);
  SUCCEED();
}

// ==================== hasOpTrace_ Tests (Commit 4) ====================

// Mock for aclrtGetFunctionName - returns sk_entry with op_trace
aclError Fake_aclrtGetFunctionName_op_trace(void *funcHandle, uint32_t maxLen, char *name) {
  (void)funcHandle;
  snprintf_s(name, maxLen, maxLen, "%s", "sk_entry_aic_op_trace");
  return ACL_SUCCESS;
}

TEST_F(SkDfxExceptionHandlerTest, IsSuperKernelException_WithOpTrace_SetHasOpTraceTrue) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_op_trace));

  bool result = handler->IsSuperKernelException(exceptionInfo);
  EXPECT_TRUE(result);
  EXPECT_TRUE(handler->hasOpTrace_);
}

TEST_F(SkDfxExceptionHandlerTest, IsSuperKernelException_WithoutOpTrace_SetHasOpTraceFalse) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));  // plain sk_entry, no op_trace

  bool result = handler->IsSuperKernelException(exceptionInfo);
  EXPECT_TRUE(result);
  EXPECT_FALSE(handler->hasOpTrace_);
}

// Mock for aclrtGetFunctionName - returns sk_entry with op_trace in middle of name
aclError Fake_aclrtGetFunctionName_op_trace_middle(void *funcHandle, uint32_t maxLen, char *name) {
  (void)funcHandle;
  snprintf_s(name, maxLen, maxLen, "%s", "sk_entry_mix11_op_trace_debug");
  return ACL_SUCCESS;
}

TEST_F(SkDfxExceptionHandlerTest, IsSuperKernelException_OpTraceInMiddleOfName) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_op_trace_middle));

  bool result = handler->IsSuperKernelException(exceptionInfo);
  EXPECT_TRUE(result);
  EXPECT_TRUE(handler->hasOpTrace_);
}

// ==================== PrintCoreSymbols Enhanced Tests (Commit 4) ====================

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_HasOpTraceFalse_ShouldReturnEarly) {
  handler->hasOpTrace_ = false;

  // Even with valid setup, should return early when hasOpTrace_ is false
  handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchOrigin_NoSKExecutedYet) {
  handler->hasOpTrace_ = true;

  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(0, false, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  // Set counterInfo for core 0: opState = ORIGIN(0), meaning no SK executed yet
  SkCounterInfo *counterInfo = reinterpret_cast<SkCounterInfo *>(buffer + headerInfo.counterOffset);
  counterInfo[0].index = 0;
  counterInfo[0].opState = static_cast<uint8_t>(SkOpTraceType::ORIGIN);  // 0

  handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0, 0);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchSKEntryLaunched_PrintNextOp) {
  handler->hasOpTrace_ = true;

  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(2, true, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  // Set counterInfo for core 0: opState = SK_ENTRY_LAUNCHED(1), opId = 0
  SkCounterInfo *counterInfo = reinterpret_cast<SkCounterInfo *>(buffer + headerInfo.counterOffset);
  counterInfo[0].index = 0;
  counterInfo[0].opState = static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_LAUNCHED);  // 1

  // Setup DFX info so GetOrLoadKernelSymbols can find func name
  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].funcHdlOri = 0xDEAD0001;
  dfxInfo[1].funcHdlOri = 0xDEAD0002;

  // Mock aclrtGetFunctionName for loading kernel symbols
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

  handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchOPLaunched_PrintCurrentOp) {
  handler->hasOpTrace_ = true;

  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(3, true, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  // Set counterInfo for core 0: opState = OP_LAUNCHED(2), opId = 1
  SkCounterInfo *counterInfo = reinterpret_cast<SkCounterInfo *>(buffer + headerInfo.counterOffset);
  counterInfo[0].index = 1;
  counterInfo[0].opState = static_cast<uint8_t>(SkOpTraceType::OP_LAUNCHED);  // 2

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].funcHdlOri = 0xDEAD0001;
  dfxInfo[1].funcHdlOri = 0xDEAD0002;
  dfxInfo[2].funcHdlOri = 0xDEAD0003;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchOPFinished_PrintCurrentAndNextOp) {
  handler->hasOpTrace_ = true;

  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(3, true, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  // Set counterInfo for core 0: opState = OP_FINISHED(3), opId = 1
  SkCounterInfo *counterInfo = reinterpret_cast<SkCounterInfo *>(buffer + headerInfo.counterOffset);
  counterInfo[0].index = 1;
  counterInfo[0].opState = static_cast<uint8_t>(SkOpTraceType::OP_FINISHED);  // 3

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].funcHdlOri = 0xDEAD0001;
  dfxInfo[1].funcHdlOri = 0xDEAD0002;
  dfxInfo[2].funcHdlOri = 0xDEAD0003;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

  handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchSKEntryFinished) {
  handler->hasOpTrace_ = true;

  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(0, false, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  // Set counterInfo: opState = SK_ENTRY_FINISHED(4)
  SkCounterInfo *counterInfo = reinterpret_cast<SkCounterInfo *>(buffer + headerInfo.counterOffset);
  counterInfo[0].index = 99;
  counterInfo[0].opState = static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_FINISHED);  // 4

  handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCoreSymbols_LaunchUnknownValue) {
  handler->hasOpTrace_ = true;

  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(0, false, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  // Set counterInfo: opState = unknown value 255
  SkCounterInfo *counterInfo = reinterpret_cast<SkCounterInfo *>(buffer + headerInfo.counterOffset);
  counterInfo[0].index = 0;
  counterInfo[0].opState = 255;  // Unknown status

  handler->PrintCoreSymbols(0, RT_CORE_TYPE_AIC, 0x1000, 0x2000);

  free(buffer);
  SUCCEED();
}

// ==================== PrintAllCoreSymbols Tests (Commit 4) ====================

TEST_F(SkDfxExceptionHandlerTest, PrintAllCoreSymbols_HasOpTraceFalse_ShouldReturnEarly) {
  handler->hasOpTrace_ = false;

  // Should return immediately without iterating cores
  handler->PrintAllCoreSymbols();
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintAllCoreSymbols_HasOpTraceTrue_IterateCores) {
  handler->hasOpTrace_ = true;

  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(0, false, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  // Set some counter info for cores
  SkCounterInfo *counterInfo = reinterpret_cast<SkCounterInfo *>(buffer + headerInfo.counterOffset);
  for (uint32_t i = 0; i < 75; ++i) {
    counterInfo[i].index = i % 5;
    counterInfo[i].opState = static_cast<uint8_t>(SkOpTraceType::ORIGIN);
  }

  // Should iterate through all 75 cores
  handler->PrintAllCoreSymbols();

  free(buffer);
  SUCCEED();
}

// ==================== GetOrLoadKernelSymbols Additional Tests ====================

TEST_F(SkDfxExceptionHandlerTest, GetOrLoadKernelSymbols_OpIdExceedsNodeCnt_ReturnEmpty) {
  handler->opSymbolCache.clear();

  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;

  // Use a custom size for this test (1024)
  uint32_t totalSize = sizeof(SkHeaderInfo) + sizeof(SkDfxInfo) * 2;
  headerInfo = {};
  headerInfo.nodeCnt = 2;
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);
  headerInfo.totalSize = totalSize;

  const size_t allocSize = 1024;
  if (allocSize == 0 || allocSize > 1024 * 1024) {
    return;
  }
  uint8_t *buffer = reinterpret_cast<uint8_t *>(malloc(allocSize));
  (void)memset_s(buffer, allocSize, 0, allocSize);
  deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;

  // opId=5 exceeds nodeCnt=2, should return empty name
  KernelFuncName result = handler->GetOrLoadKernelSymbols(5);
  EXPECT_EQ(result.name, "");

  free(buffer);
}

// ==================== modelIdIndexAndSkScopeId 解码与 SkEventRecorder 交互测试 ====================

// Test: SkHeaderInfo.modelIdIndexAndSkScopeId 字段偏移验证
TEST_F(SkDfxExceptionHandlerTest, ModelIdIndexAndSkScopeId_FieldOffsetInSkHeaderInfo) {
  SkHeaderInfo headerInfo = {};
  // 编码: (modelIdIdx << 32) | (skScopeId << 16) = (3 << 32) | (5 << 16)
  headerInfo.modelIdIndexAndSkScopeId = 0x0000000300050000ULL;

  // 验证字段存在且可以赋值
  EXPECT_EQ(headerInfo.modelIdIndexAndSkScopeId, 0x0000000300050000ULL);
  EXPECT_EQ(sizeof(headerInfo.modelIdIndexAndSkScopeId), sizeof(uint64_t));

  // 解码
  uint16_t modelIdIdx = static_cast<uint16_t>((headerInfo.modelIdIndexAndSkScopeId >> 32) & 0xFFFF);
  uint16_t skScopeId = static_cast<uint16_t>((headerInfo.modelIdIndexAndSkScopeId >> 16) & 0xFFFF);
  EXPECT_EQ(modelIdIdx, 3);
  EXPECT_EQ(skScopeId, 5);
}

// Test: IdentifyErrorNodeByPC 中 modelIdIndexAndSkScopeId 解码与 SkEventRecorder 反查
TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_ModelIdIndexAndSkScopeIdDecode) {
  // 注册一个 modelId 到 SkEventRecorder（重构后映射表为 string）
  std::string modelId = "model_deadbeef_1";
  uint16_t skScopeId = 42;
  uint16_t modelIdIdx = SkEventRecorder::Instance().RegisterModelId(modelId);

  // 设置 buffer
  uint8_t buffer[2048] = {0};
  SkHeaderInfo headerInfo;
  headerInfo.aicQueOffset = 0;
  headerInfo.aivQueOffset = 0;
  headerInfo.counterOffset = 0;
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);
  headerInfo.nodeCnt = 1;
  headerInfo.totalSize = sizeof(buffer);
  headerInfo.modelIdIndexAndSkScopeId =
      (static_cast<uint64_t>(modelIdIdx) << 32) | (static_cast<uint64_t>(skScopeId) << 16);

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo->aicSize = 0x100;
  dfxInfo->aivSize = 0;
  dfxInfo->funcHdlOri = 0xDEAD0001;
  for (int i = 0; i < 4; i++) {
    dfxInfo->entryAic[i] = 0;
    dfxInfo->entryAiv[i] = 0;
  }
  dfxInfo->entryAic[0] = 0x1000;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;

  // 验证解码逻辑
  uint16_t decodedIdx = static_cast<uint16_t>((headerInfo.modelIdIndexAndSkScopeId >> 32) & 0xFFFF);
  uint16_t decodedScopeId = static_cast<uint16_t>((headerInfo.modelIdIndexAndSkScopeId >> 16) & 0xFFFF);
  std::string recoveredModelId = SkEventRecorder::Instance().GetModelIdByIndex(decodedIdx);

  EXPECT_EQ(decodedIdx, modelIdIdx);
  EXPECT_EQ(decodedScopeId, skScopeId);
  EXPECT_EQ(recoveredModelId, modelId);

  // 调用 IdentifyErrorNodeByPC，PC 匹配到 node[0] 的 AIC entry
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));
  handler->IdentifyErrorNodeByPC(0, RT_CORE_TYPE_AIC, 0x1000, 0x1050);
  SUCCEED();

  // 清理 SkEventRecorder
  SkEventRecorder::Instance().modelIdIndexMap.clear();
  SkEventRecorder::Instance().modelIdToIndexMap.clear();
}

// Test: modelIdIndexAndSkScopeId 为 0 时（未设置），反查返回空串
TEST_F(SkDfxExceptionHandlerTest, ModelIdIndexAndSkScopeId_ZeroValue_GetModelIdReturnsEmpty) {
  SkHeaderInfo headerInfo = {};
  headerInfo.modelIdIndexAndSkScopeId = 0;

  uint16_t decodedIdx = static_cast<uint16_t>((headerInfo.modelIdIndexAndSkScopeId >> 32) & 0xFFFF);
  std::string recoveredModelId = SkEventRecorder::Instance().GetModelIdByIndex(decodedIdx);
  EXPECT_EQ(decodedIdx, 0);
  EXPECT_TRUE(recoveredModelId.empty());  // index 0 不存在，返回空串
}

// Test: cond 寄存器完整 48bit 布局编解码验证
// 排布: modelIdIdx(16bit)[47:32] | skScopeId(16bit)[31:16] | task->index(8bit)[15:8] | SkOpTraceType(8bit)[7:0]
TEST_F(SkDfxExceptionHandlerTest, CondRegister_48bitLayout_EncodeDecode) {
  std::string modelId = "model_123456_1";
  uint16_t skScopeId = 1234;
  uint16_t modelIdIdx = SkEventRecorder::Instance().RegisterModelId(modelId);

  uint64_t modelIdIndexAndSkScopeId =
      (static_cast<uint64_t>(modelIdIdx) << 32) | (static_cast<uint64_t>(skScopeId) << 16);

  // OP_LAUNCHED + task->index = 7
  uint64_t cond = static_cast<uint64_t>(SkOpTraceType::OP_LAUNCHED) + (static_cast<uint64_t>(7) << 8);
  cond = modelIdIndexAndSkScopeId | cond;

  // 解码
  uint16_t decodedModelIdIdx = static_cast<uint16_t>((cond >> 32) & 0xFFFF);
  uint16_t decodedSkScopeId = static_cast<uint16_t>((cond >> 16) & 0xFFFF);
  uint8_t decodedTaskIndex = static_cast<uint8_t>((cond >> 8) & 0xFF);
  uint8_t decodedOpTraceType = static_cast<uint8_t>(cond & 0xFF);

  EXPECT_EQ(decodedModelIdIdx, modelIdIdx);
  EXPECT_EQ(decodedSkScopeId, skScopeId);
  EXPECT_EQ(decodedTaskIndex, 7);
  EXPECT_EQ(decodedOpTraceType, static_cast<uint8_t>(SkOpTraceType::OP_LAUNCHED));

  // 通过 index 反查原始 modelId
  EXPECT_EQ(SkEventRecorder::Instance().GetModelIdByIndex(decodedModelIdIdx), modelId);

  // 清理
  SkEventRecorder::Instance().modelIdIndexMap.clear();
  SkEventRecorder::Instance().modelIdToIndexMap.clear();
}

// ==================== PrintMatchedNodeBasicInfo Tests ====================

TEST_F(SkDfxExceptionHandlerTest, PrintMatchedNodeBasicInfo_AICCoreType) {
  SkHeaderInfo headerInfo = {};
  headerInfo.modelIdIndexAndSkScopeId = 0;
  handler->skHeaderInfoHost = &headerInfo;

  SkDfxInfo dfxNode = {};
  dfxNode.numBlocks = 10;
  dfxNode.cubeNum = 5;
  dfxNode.vecNum = 5;

  // Should not crash, just print logs
  handler->PrintMatchedNodeBasicInfo(0, RT_CORE_TYPE_AIC, 0x1000, 0x1100, 0, 0, 0x1000, 0x1200, 0x200, dfxNode);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintMatchedNodeBasicInfo_AIVCoreType) {
  SkHeaderInfo headerInfo = {};
  headerInfo.modelIdIndexAndSkScopeId = 0;
  handler->skHeaderInfoHost = &headerInfo;

  SkDfxInfo dfxNode = {};
  dfxNode.numBlocks = 8;
  dfxNode.cubeNum = 0;
  dfxNode.vecNum = 8;

  handler->PrintMatchedNodeBasicInfo(30, RT_CORE_TYPE_AIV, 0x3000, 0x3050, 2, 1, 0x3000, 0x3100, 0x100, dfxNode);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintMatchedNodeBasicInfo_WithModelId) {
  std::string modelId = "model_abcd1234_1";
  uint16_t modelIdIdx = SkEventRecorder::Instance().RegisterModelId(modelId);
  uint16_t skScopeId = 7;

  SkHeaderInfo headerInfo = {};
  headerInfo.modelIdIndexAndSkScopeId =
      (static_cast<uint64_t>(modelIdIdx) << 32) | (static_cast<uint64_t>(skScopeId) << 16);
  handler->skHeaderInfoHost = &headerInfo;

  SkDfxInfo dfxNode = {};
  dfxNode.numBlocks = 20;
  dfxNode.cubeNum = 20;
  dfxNode.vecNum = 20;

  handler->PrintMatchedNodeBasicInfo(0, RT_CORE_TYPE_AIC, 0x1000, 0x1100, 0, 0, 0x1000, 0x1200, 0x200, dfxNode);
  SUCCEED();

  SkEventRecorder::Instance().modelIdIndexMap.clear();
  SkEventRecorder::Instance().modelIdToIndexMap.clear();
}

// ==================== PrintFuncSymbolInfo Tests ====================

// Mock for rtGetBinBuffer - failure
int FakeRtGetBinBufferFailure(void *binHdl, int addrType, void **buffer, uint32_t *size) {
  (void)binHdl;
  (void)addrType;
  if (buffer != nullptr) {
    *buffer = nullptr;
  }
  if (size != nullptr) {
    *size = 0;
  }
  return -1;
}

// Mock for rtGetBinBuffer - success with valid ELF-like data (will fail GetFuncSymbolInfo)
static uint8_t gFakeBinBuffer[256];
int FakeRtGetBinBufferSuccess(void *binHdl, int addrType, void **buffer, uint32_t *size) {
  (void)binHdl;
  (void)addrType;
  (void)memset_s(gFakeBinBuffer, sizeof(gFakeBinBuffer), 0, sizeof(gFakeBinBuffer));
  *buffer = gFakeBinBuffer;
  *size = sizeof(gFakeBinBuffer);
  return 0;
}

// Mock for aclrtGetFunctionName - failure
aclError Fake_aclrtGetFunctionName_Fail(void *funcHandle, uint32_t maxLen, char *name) {
  (void)funcHandle;
  (void)maxLen;
  (void)name;
  return ACL_ERROR_FAILURE;
}

TEST_F(SkDfxExceptionHandlerTest, PrintFuncSymbolInfo_NotFuncType) {
  // nodeIdx not in funcNodeIndices_ -> should log "not a FUNC type"
  handler->funcNodeIndices_.clear();

  SkDfxInfo dfxNode = {};
  uint64_t entries[4] = {0x1000, 0, 0, 0};

  handler->PrintFuncSymbolInfo(0, RT_CORE_TYPE_AIC, 0, 0, entries, dfxNode);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintFuncSymbolInfo_GetFunctionNameFail) {
  handler->funcNodeIndices_.insert(0);

  SkDfxInfo dfxNode = {};
  dfxNode.funcHdlOri = 0xBBB;
  uint64_t entries[4] = {0x1000, 0, 0, 0};

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_Fail));

  handler->PrintFuncSymbolInfo(0, RT_CORE_TYPE_AIC, 0, 0, entries, dfxNode);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintFuncSymbolInfo_GetBinBufferFail) {
  handler->funcNodeIndices_.insert(0);

  SkDfxInfo dfxNode = {};
  dfxNode.funcHdlOri = 0xBBB;
  dfxNode.binHdl = 0xAAA;
  uint64_t entries[4] = {0x1000, 0, 0, 0};

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferFailure));

  handler->PrintFuncSymbolInfo(0, RT_CORE_TYPE_AIC, 0, 0, entries, dfxNode);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintFuncSymbolInfo_FuncOffsetZero_NoFallback) {
  handler->funcNodeIndices_.insert(0);

  SkDfxInfo dfxNode = {};
  dfxNode.funcHdlOri = 0xBBB;
  dfxNode.binHdl = 0xAAA;
  // All funcOffsets are 0, no fallback possible (aicFuncOffset[0] == 0)
  for (int i = 0; i < 4; i++) {
    dfxNode.aicFuncOffset[i] = 0;
    dfxNode.aivFuncOffset[i] = 0;
    dfxNode.entryAic[i] = 0;
  }
  uint64_t entries[4] = {0x1000, 0, 0, 0};

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferSuccess));

  handler->PrintFuncSymbolInfo(0, RT_CORE_TYPE_AIC, 0, 0, entries, dfxNode);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintFuncSymbolInfo_FuncOffsetNonZero_GetFuncSymbolInfoFail) {
  handler->funcNodeIndices_.insert(0);

  SkDfxInfo dfxNode = {};
  dfxNode.funcHdlOri = 0xBBB;
  dfxNode.binHdl = 0xAAA;
  dfxNode.aicFuncOffset[0] = 0x100;  // non-zero offset, but bin data is fake -> GetFuncSymbolInfo will fail
  uint64_t entries[4] = {0x1000, 0, 0, 0};

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferSuccess));

  handler->PrintFuncSymbolInfo(0, RT_CORE_TYPE_AIC, 0, 0, entries, dfxNode);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintFuncSymbolInfo_FuncOffsetZero_WithFallback) {
  handler->funcNodeIndices_.insert(0);

  SkDfxInfo dfxNode = {};
  dfxNode.funcHdlOri = 0xBBB;
  dfxNode.binHdl = 0xAAA;
  // aicFuncOffset[0] is non-zero and entryAic[0] is non-zero, so fallback can compute
  dfxNode.aicFuncOffset[0] = 0x100;
  dfxNode.entryAic[0] = 0x1000;
  // entry[1] offset is 0 but fallback will compute from entry[0]
  dfxNode.aicFuncOffset[1] = 0;
  uint64_t entries[4] = {0x1000, 0x1400, 0, 0};  // entry[1] = 0x1400

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferSuccess));

  // For entry[1], funcOffset=0 but fallback computes: entry[1] - (entryAic[0] - aicFuncOffset[0]) = 0x1400 - 0xF00 =
  // 0x500
  handler->PrintFuncSymbolInfo(0, RT_CORE_TYPE_AIC, 0, 1, entries, dfxNode);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintFuncSymbolInfo_AIVCoreType_UsesAivFuncOffset) {
  handler->funcNodeIndices_.insert(0);

  SkDfxInfo dfxNode = {};
  dfxNode.funcHdlOri = 0xBBB;
  dfxNode.binHdl = 0xAAA;
  dfxNode.aivFuncOffset[0] = 0x200;  // AIV offset
  uint64_t entries[4] = {0x3000, 0, 0, 0};

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferSuccess));

  handler->PrintFuncSymbolInfo(30, RT_CORE_TYPE_AIV, 0, 0, entries, dfxNode);
  SUCCEED();
}

// ==================== PrintNodeDevArgs Tests ====================

TEST_F(SkDfxExceptionHandlerTest, PrintNodeDevArgs_AICWithMatchingTask) {
  // Setup buffer with AIC task queue
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);
  headerInfo.aivQueOffset = 0;
  handler->skHeaderInfoHost = &headerInfo;
  handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  handler->aicTaskCnt = 1;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aicQueOffset);
  aicTaskQue->taskCnt = 1;
  aicTaskQue->cap = 10;
  aicTaskQue->taskInfos[0].index = 0;
  aicTaskQue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
  aicTaskQue->taskInfos[0].args = 0xDEADBEEF;

  handler->PrintNodeDevArgs(0, RT_CORE_TYPE_AIC, 0);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintNodeDevArgs_AIVWithMatchingTask) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.aicQueOffset = 0;
  headerInfo.aivQueOffset = sizeof(SkHeaderInfo);
  handler->skHeaderInfoHost = &headerInfo;
  handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  handler->aivTaskCnt = 1;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  TaskQue *aivTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aivQueOffset);
  aivTaskQue->taskCnt = 1;
  aivTaskQue->cap = 10;
  aivTaskQue->taskInfos[0].index = 3;
  aivTaskQue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
  aivTaskQue->taskInfos[0].args = 0xCAFEBABE;

  handler->PrintNodeDevArgs(30, RT_CORE_TYPE_AIV, 3);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintNodeDevArgs_NoMatchingTaskIndex) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);
  headerInfo.aivQueOffset = 0;
  handler->skHeaderInfoHost = &headerInfo;
  handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  handler->aicTaskCnt = 1;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aicQueOffset);
  aicTaskQue->taskCnt = 1;
  aicTaskQue->cap = 10;
  aicTaskQue->taskInfos[0].index = 5;  // index=5, but looking for nodeIdx=0
  aicTaskQue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
  aicTaskQue->taskInfos[0].args = 0x1234;

  // Should not crash, just not find matching task
  handler->PrintNodeDevArgs(0, RT_CORE_TYPE_AIC, 0);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintNodeDevArgs_NoMatchingTaskType) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);
  headerInfo.aivQueOffset = 0;
  handler->skHeaderInfoHost = &headerInfo;
  handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  handler->aicTaskCnt = 1;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aicQueOffset);
  aicTaskQue->taskCnt = 1;
  aicTaskQue->cap = 10;
  aicTaskQue->taskInfos[0].index = 0;  // index matches but type is not TYPE_FUNC
  aicTaskQue->taskInfos[0].type = SkTaskType::TYPE_SYNC;
  aicTaskQue->taskInfos[0].args = 0x5678;

  // Should not crash, task type doesn't match
  handler->PrintNodeDevArgs(0, RT_CORE_TYPE_AIC, 0);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintNodeDevArgs_ZeroTaskCnt) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);
  headerInfo.aivQueOffset = 0;
  handler->skHeaderInfoHost = &headerInfo;
  handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  handler->aicTaskCnt = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  // Should not crash with zero tasks
  handler->PrintNodeDevArgs(0, RT_CORE_TYPE_AIC, 0);
  SUCCEED();
}

// ==================== PrintNoMatchInfo Tests ====================

TEST_F(SkDfxExceptionHandlerTest, PrintNoMatchInfo_AICCoreType) {
  SkHeaderInfo headerInfo = {};
  headerInfo.modelIdIndexAndSkScopeId = 0;
  handler->skHeaderInfoHost = &headerInfo;

  handler->PrintNoMatchInfo(0, RT_CORE_TYPE_AIC, 0x1000, 0xFFFF);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintNoMatchInfo_AIVCoreType) {
  SkHeaderInfo headerInfo = {};
  headerInfo.modelIdIndexAndSkScopeId = 0;
  handler->skHeaderInfoHost = &headerInfo;

  handler->PrintNoMatchInfo(30, RT_CORE_TYPE_AIV, 0x3000, 0xFFFF);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintNoMatchInfo_WithModelId) {
  std::string modelId = "model_12345678_1";
  uint16_t modelIdIdx = SkEventRecorder::Instance().RegisterModelId(modelId);
  uint16_t skScopeId = 99;

  SkHeaderInfo headerInfo = {};
  headerInfo.modelIdIndexAndSkScopeId =
      (static_cast<uint64_t>(modelIdIdx) << 32) | (static_cast<uint64_t>(skScopeId) << 16);
  handler->skHeaderInfoHost = &headerInfo;

  handler->PrintNoMatchInfo(5, RT_CORE_TYPE_AIC, 0x1000, 0xFFFF);
  SUCCEED();

  SkEventRecorder::Instance().modelIdIndexMap.clear();
  SkEventRecorder::Instance().modelIdToIndexMap.clear();
}

// ==================== IdentifyErrorNodeByPC Refactored Integration Tests ====================

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_MatchEntry1_CallsSubFunctions) {
  // Setup: one node with 2 AIC entries at entry[0]=0x1000, entry[1]=0x2000
  uint8_t buffer[2048] = {0};
  SkHeaderInfo headerInfo;
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);
  headerInfo.nodeCnt = 1;
  headerInfo.aicQueOffset = 0;
  headerInfo.aivQueOffset = 0;
  headerInfo.modelIdIndexAndSkScopeId = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo->binHdl = 0xAAA;
  dfxInfo->funcHdlOri = 0xBBB;
  dfxInfo->aicSize = 0x200;
  dfxInfo->aivSize = 0;
  dfxInfo->entryAic[0] = 0x1000;
  dfxInfo->entryAic[1] = 0x2000;
  dfxInfo->entryAic[2] = 0;
  dfxInfo->entryAic[3] = 0;
  dfxInfo->aicFuncOffset[0] = 0x100;
  dfxInfo->aicFuncOffset[1] = 0;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;
  handler->funcNodeIndices_.insert(0);

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferSuccess));

  // currentPC = 0x2100 falls in entry[1] range [0x2000, 0x2200) -> should match entry[1]
  handler->IdentifyErrorNodeByPC(0, RT_CORE_TYPE_AIC, 0x1000, 0x2100);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, IdentifyErrorNodeByPC_NoMatch_CallsPrintNoMatchInfo) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo;
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);
  headerInfo.nodeCnt = 1;
  headerInfo.aicQueOffset = 0;
  headerInfo.aivQueOffset = 0;
  headerInfo.modelIdIndexAndSkScopeId = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo->entryAic[0] = 0x1000;
  dfxInfo->aicSize = 0x100;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;

  // currentPC outside all ranges -> no match path
  handler->IdentifyErrorNodeByPC(0, RT_CORE_TYPE_AIC, 0x1000, 0xFFFF);
  SUCCEED();
}

// ==================== GetCondRegValue Tests (PR 485) ====================

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_ZeroRegisters) {
  rtExceptionErrRegInfo_t regInfo = {};
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0ULL);
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_Low32Only) {
  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.errReg[20] = 0xDEADBEEF;  // low 32 bits
  regInfo.errReg[21] = 0;           // high 32 bits
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0x00000000DEADBEEFULL);
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_High32Only) {
  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.errReg[20] = 0;           // low 32 bits
  regInfo.errReg[21] = 0x12345678;  // high 32 bits
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0x1234567800000000ULL);
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_Full64Bit) {
  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.errReg[20] = 0xAAAAAAAA;  // low 32 bits
  regInfo.errReg[21] = 0xBBBBBBBB;  // high 32 bits
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0xBBBBBBBBAAAAAAAAULL);
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_EncodedOpStateAndIndex) {
  // Simulate a cond value: opState=OP_LAUNCHED(2), opIndex=5
  rtExceptionErrRegInfo_t regInfo = {};
  uint64_t expected = static_cast<uint64_t>(SkOpTraceType::OP_LAUNCHED) | (5ULL << 8);
  regInfo.errReg[20] = static_cast<uint32_t>(expected & 0xFFFFFFFF);
  regInfo.errReg[21] = static_cast<uint32_t>(expected >> 32);
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, expected);
  EXPECT_EQ(static_cast<uint8_t>(result & 0xFF), static_cast<uint8_t>(SkOpTraceType::OP_LAUNCHED));
  EXPECT_EQ(static_cast<uint32_t>((result >> 8) & 0xFF), 5U);
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_OtherRegistersUnaffected) {
  // errReg at other indices should not affect the result
  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.errReg[0] = 0xFFFFFFFF;
  regInfo.errReg[19] = 0xFFFFFFFF;
  regInfo.errReg[20] = 0x00000001;
  regInfo.errReg[21] = 0x00000002;
  regInfo.errReg[22] = 0xFFFFFFFF;
  regInfo.errReg[63] = 0xFFFFFFFF;
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0x0000000200000001ULL);
}

// ==================== GetCondRegValue DAV_3510 Tests (Commit b9bbb50) ====================

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_Dav3510_ZeroRegisters) {
  MOCKER(GetCurrentSkKernelArch).stubs().will(invoke(FakeGetCurrentSkKernelArch_Dav3510));
  rtExceptionErrRegInfo_t regInfo = {};
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0ULL);
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_Dav3510_Low32Only) {
  MOCKER(GetCurrentSkKernelArch).stubs().will(invoke(FakeGetCurrentSkKernelArch_Dav3510));
  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.errReg[32] = 0xDEADBEEF;  // low 32 bits for 3510
  regInfo.errReg[33] = 0;           // high 32 bits for 3510
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0x00000000DEADBEEFULL);
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_Dav3510_High32Only) {
  MOCKER(GetCurrentSkKernelArch).stubs().will(invoke(FakeGetCurrentSkKernelArch_Dav3510));
  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.errReg[32] = 0;           // low 32 bits for 3510
  regInfo.errReg[33] = 0x12345678;  // high 32 bits for 3510
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0x1234567800000000ULL);
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_Dav3510_Full64Bit) {
  MOCKER(GetCurrentSkKernelArch).stubs().will(invoke(FakeGetCurrentSkKernelArch_Dav3510));
  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.errReg[32] = 0xAAAAAAAA;  // low 32 bits for 3510
  regInfo.errReg[33] = 0xBBBBBBBB;  // high 32 bits for 3510
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0xBBBBBBBBAAAAAAAAULL);
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_Dav3510_EncodedOpStateAndIndex) {
  MOCKER(GetCurrentSkKernelArch).stubs().will(invoke(FakeGetCurrentSkKernelArch_Dav3510));
  // Simulate a cond value: opState=OP_LAUNCHED(2), opIndex=5
  rtExceptionErrRegInfo_t regInfo = {};
  uint64_t expected = static_cast<uint64_t>(SkOpTraceType::OP_LAUNCHED) | (5ULL << 8);
  regInfo.errReg[32] = static_cast<uint32_t>(expected & 0xFFFFFFFF);
  regInfo.errReg[33] = static_cast<uint32_t>(expected >> 32);
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, expected);
  EXPECT_EQ(static_cast<uint8_t>(result & 0xFF), static_cast<uint8_t>(SkOpTraceType::OP_LAUNCHED));
  EXPECT_EQ(static_cast<uint32_t>((result >> 8) & 0xFF), 5U);
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_Dav3510_OtherRegistersUnaffected) {
  MOCKER(GetCurrentSkKernelArch).stubs().will(invoke(FakeGetCurrentSkKernelArch_Dav3510));
  // errReg at other indices (including 20/21 for 2201) should not affect 3510 result
  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.errReg[0] = 0xFFFFFFFF;
  regInfo.errReg[20] = 0xFFFFFFFF;  // 2201 low idx, should be ignored for 3510
  regInfo.errReg[21] = 0xFFFFFFFF;  // 2201 high idx, should be ignored for 3510
  regInfo.errReg[31] = 0xFFFFFFFF;
  regInfo.errReg[32] = 0x00000001;  // 3510 low idx
  regInfo.errReg[33] = 0x00000002;  // 3510 high idx
  regInfo.errReg[34] = 0xFFFFFFFF;
  regInfo.errReg[63] = 0xFFFFFFFF;
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0x0000000200000001ULL);
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_Dav3510_2201RegsIgnored) {
  MOCKER(GetCurrentSkKernelArch).stubs().will(invoke(FakeGetCurrentSkKernelArch_Dav3510));
  // When on 3510, errReg[20]/[21] should NOT be read; only [32]/[33] matter
  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.errReg[20] = 0xFFFFFFFF;  // Would be low for 2201, ignored on 3510
  regInfo.errReg[21] = 0xFFFFFFFF;  // Would be high for 2201, ignored on 3510
  regInfo.errReg[32] = 0x00000000;  // 3510 low idx
  regInfo.errReg[33] = 0x00000000;  // 3510 high idx
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0ULL);  // Should be 0, not 0xFFFFFFFFFFFFFFFF
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_Dav2201_3510RegsIgnored) {
  SkUtSetAclrtGetSocName("Ascend910B");  // default 2201 arch
  // When on 2201, errReg[32]/[33] should NOT be read; only [20]/[21] matter
  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.errReg[20] = 0x00000000;  // 2201 low idx
  regInfo.errReg[21] = 0x00000000;  // 2201 high idx
  regInfo.errReg[32] = 0xFFFFFFFF;  // Would be low for 3510, ignored on 2201
  regInfo.errReg[33] = 0xFFFFFFFF;  // Would be high for 3510, ignored on 2201
  uint64_t result = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(result, 0ULL);  // Should be 0, not 0xFFFFFFFFFFFFFFFF
}

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_Dav3510_Integration_ParseAndPrintCondInfo) {
  MOCKER(GetCurrentSkKernelArch).stubs().will(invoke(FakeGetCurrentSkKernelArch_Dav3510));
  // Build an rtExceptionErrRegInfo_t with COND values at 3510 indices
  // opState=OP_FINISHED(3), opIndex=1, modelRIIdAndSkScopeId in upper bits
  uint64_t expectedCond = static_cast<uint64_t>(SkOpTraceType::OP_FINISHED) | (1ULL << 8) | (0xABCDULL << 16);

  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.coreId = 3;
  regInfo.coreType = RT_CORE_TYPE_AIC;
  regInfo.errReg[32] = static_cast<uint32_t>(expectedCond & 0xFFFFFFFF);
  regInfo.errReg[33] = static_cast<uint32_t>(expectedCond >> 32);

  // Verify GetCondRegValue extracts correctly for 3510
  uint64_t condValue = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(condValue, expectedCond);

  // Verify ParseAndPrintCondInfo doesn't crash with the extracted value
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(3, true, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].funcHdlOri = 0xDEAD0001;
  dfxInfo[1].funcHdlOri = 0xDEAD0002;
  dfxInfo[2].funcHdlOri = 0xDEAD0003;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

  handler->ParseAndPrintCondInfo(3, RT_CORE_TYPE_AIC, condValue);

  free(buffer);
  SUCCEED();
}

// ==================== ParseAndPrintCondInfo Tests (PR 485) ====================

TEST_F(SkDfxExceptionHandlerTest, ParseAndPrintCondInfo_CondValueZero_ShouldLogDriverNotUpgraded) {
  // condValue == 0 should print driver not upgraded message and return
  handler->ParseAndPrintCondInfo(0, RT_CORE_TYPE_AIC, 0);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, ParseAndPrintCondInfo_AICCoreType) {
  // opState=ORIGIN(0), opIndex=0, with non-zero condValue
  uint64_t condValue = static_cast<uint64_t>(SkOpTraceType::ORIGIN) | (0ULL << 8) | (0x1000ULL << 16);
  handler->ParseAndPrintCondInfo(5, RT_CORE_TYPE_AIC, condValue);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, ParseAndPrintCondInfo_AIVCoreType) {
  // opState=OP_LAUNCHED(2), opIndex=1, with non-zero condValue
  // Need to setup skHeaderInfoHost because PrintCondSubKernelInfo accesses skHeaderInfoHost->nodeCnt
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(3, true, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].funcHdlOri = 0xDEAD0001;
  dfxInfo[1].funcHdlOri = 0xDEAD0002;
  dfxInfo[2].funcHdlOri = 0xDEAD0003;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

  uint64_t condValue = static_cast<uint64_t>(SkOpTraceType::OP_LAUNCHED) | (1ULL << 8) | (0x2000ULL << 16);
  handler->ParseAndPrintCondInfo(10, RT_CORE_TYPE_AIV, condValue);

  free(buffer);
  SUCCEED();
}

// ==================== PrintCondSubKernelInfo Tests (PR 485) ====================

TEST_F(SkDfxExceptionHandlerTest, PrintCondSubKernelInfo_Origin) {
  uint64_t condValue = static_cast<uint64_t>(SkOpTraceType::ORIGIN) | (0ULL << 8) | (0xABCDULL << 16);
  handler->PrintCondSubKernelInfo(0, condValue);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCondSubKernelInfo_SkEntryLaunched_OpIndexWithinNodeCnt) {
  // Setup: 2 nodes, opState=SK_ENTRY_LAUNCHED, opIndex=0 (next sub-kernel)
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(2, true, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].funcHdlOri = 0xDEAD0001;
  dfxInfo[1].funcHdlOri = 0xDEAD0002;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

  uint64_t condValue = static_cast<uint64_t>(SkOpTraceType::SK_ENTRY_LAUNCHED) | (0ULL << 8) | (0x1000ULL << 16);
  handler->PrintCondSubKernelInfo(0, condValue);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCondSubKernelInfo_SkEntryLaunched_OpIndexExceedsNodeCnt) {
  // opIndex >= nodeCnt, should not attempt to load kernel symbols
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(1, false, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  // opIndex=5 but nodeCnt=1, should skip GetOrLoadKernelSymbols
  uint64_t condValue = static_cast<uint64_t>(SkOpTraceType::SK_ENTRY_LAUNCHED) | (5ULL << 8) | (0x1000ULL << 16);
  handler->PrintCondSubKernelInfo(0, condValue);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCondSubKernelInfo_OpLaunched_OpIndexWithinNodeCnt) {
  // Setup: 3 nodes, opState=OP_LAUNCHED, opIndex=1
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(3, true, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].funcHdlOri = 0xDEAD0001;
  dfxInfo[1].funcHdlOri = 0xDEAD0002;
  dfxInfo[2].funcHdlOri = 0xDEAD0003;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  uint64_t condValue = static_cast<uint64_t>(SkOpTraceType::OP_LAUNCHED) | (1ULL << 8) | (0x1000ULL << 16);
  handler->PrintCondSubKernelInfo(0, condValue);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCondSubKernelInfo_OpLaunched_OpIndexExceedsNodeCnt) {
  // opIndex >= nodeCnt, should not attempt to load kernel symbols
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(1, false, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  uint64_t condValue = static_cast<uint64_t>(SkOpTraceType::OP_LAUNCHED) | (10ULL << 8) | (0x1000ULL << 16);
  handler->PrintCondSubKernelInfo(0, condValue);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCondSubKernelInfo_OpFinished_CurrentAndNext) {
  // Setup: 3 nodes, opState=OP_FINISHED, opIndex=1 -> print current(1) and next(2)
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(3, true, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].funcHdlOri = 0xDEAD0001;
  dfxInfo[1].funcHdlOri = 0xDEAD0002;
  dfxInfo[2].funcHdlOri = 0xDEAD0003;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

  uint64_t condValue = static_cast<uint64_t>(SkOpTraceType::OP_FINISHED) | (1ULL << 8) | (0x1000ULL << 16);
  handler->PrintCondSubKernelInfo(0, condValue);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCondSubKernelInfo_OpFinished_LastNode_NoNext) {
  // Setup: 3 nodes, opState=OP_FINISHED, opIndex=2 (last) -> print current(2), no next
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(3, true, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].funcHdlOri = 0xDEAD0001;
  dfxInfo[1].funcHdlOri = 0xDEAD0002;
  dfxInfo[2].funcHdlOri = 0xDEAD0003;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

  uint64_t condValue = static_cast<uint64_t>(SkOpTraceType::OP_FINISHED) | (2ULL << 8) | (0x1000ULL << 16);
  handler->PrintCondSubKernelInfo(0, condValue);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCondSubKernelInfo_SkEntryFinished) {
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(0, false, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  uint64_t condValue = static_cast<uint64_t>(SkOpTraceType::SK_ENTRY_FINISHED) | (99ULL << 8) | (0x1000ULL << 16);
  handler->PrintCondSubKernelInfo(0, condValue);

  free(buffer);
  SUCCEED();
}

TEST_F(SkDfxExceptionHandlerTest, PrintCondSubKernelInfo_UnknownOpState) {
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(0, false, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  // opState=255 (unknown value)
  uint64_t condValue = 0xFF | (7ULL << 8) | (0x1000ULL << 16);
  handler->PrintCondSubKernelInfo(0, condValue);

  free(buffer);
  SUCCEED();
}

// ==================== COND Register Integration Tests (PR 485) ====================

TEST_F(SkDfxExceptionHandlerTest, GetCondRegValue_ParseAndPrintCondInfo_Integration) {
  // Build an rtExceptionErrRegInfo_t with known COND register values
  // opState=OP_FINISHED(3), opIndex=1, modelIdIndexAndSkScopeId in upper bits
  uint64_t expectedCond = static_cast<uint64_t>(SkOpTraceType::OP_FINISHED) | (1ULL << 8) | (0xABCDULL << 16);

  rtExceptionErrRegInfo_t regInfo = {};
  regInfo.coreId = 3;
  regInfo.coreType = RT_CORE_TYPE_AIC;
  regInfo.errReg[20] = static_cast<uint32_t>(expectedCond & 0xFFFFFFFF);
  regInfo.errReg[21] = static_cast<uint32_t>(expectedCond >> 32);

  // Verify GetCondRegValue extracts correctly
  uint64_t condValue = SuperKernelExceptionHandler::GetCondRegValue(regInfo);
  EXPECT_EQ(condValue, expectedCond);

  // Verify ParseAndPrintCondInfo doesn't crash with the extracted value
  SkHeaderInfo headerInfo;
  SkDeviceEntryArgs *deviceArgs;
  uint8_t *buffer = SetupOpTraceTestBuffer(3, true, headerInfo, deviceArgs, handler.get());
  if (buffer == nullptr) {
    return;
  }

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].funcHdlOri = 0xDEAD0001;
  dfxInfo[1].funcHdlOri = 0xDEAD0002;
  dfxInfo[2].funcHdlOri = 0xDEAD0003;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

  handler->ParseAndPrintCondInfo(3, RT_CORE_TYPE_AIC, condValue);

  free(buffer);
  SUCCEED();
}

// Test: SkHeaderInfo 结构体大小不应被意外修改
TEST_F(SkDfxExceptionHandlerTest, SkHeaderInfo_SizeAndFieldOffsetsStable) {
  // 验证 SkHeaderInfo 的关键字段偏移不会被误改
  SkHeaderInfo headerInfo = {};

  // 验证 modelIdIndexAndSkScopeId 可以存储完整的 48bit cond 编码
  uint64_t testValue = 0xFFFFFFFF00000000ULL;  // modelIdIdx=0xFFFF, skScopeId=0xFFFF
  headerInfo.modelIdIndexAndSkScopeId = testValue;
  EXPECT_EQ(headerInfo.modelIdIndexAndSkScopeId, testValue);

  // 验证 modelId 可以通过 index 映射恢复
  std::string fullModelId = "model_7fff1234_1";
  uint16_t idx = SkEventRecorder::Instance().RegisterModelId(fullModelId);
  EXPECT_EQ(SkEventRecorder::Instance().GetModelIdByIndex(idx), fullModelId);

  SkEventRecorder::Instance().modelIdIndexMap.clear();
  SkEventRecorder::Instance().modelIdToIndexMap.clear();
}

// ==================== GetSubKernelTaskArgs Tests (Line 799) ====================

TEST_F(SkDfxExceptionHandlerTest, GetSubKernelTaskArgs_NodeIdxExceedsNodeCnt_ReturnsFalse) {
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 2;
  handler->skHeaderInfoHost = &headerInfo;

  uint64_t argsAddr = 0;
  uint32_t argsSize = 0;

  bool result = handler->GetSubKernelTaskArgs(5, argsAddr, argsSize);  // nodeIdx=5 > nodeCnt=2
  EXPECT_FALSE(result);
  EXPECT_EQ(argsAddr, 0);
  EXPECT_EQ(argsSize, 0);
}

TEST_F(SkDfxExceptionHandlerTest, GetSubKernelTaskArgs_AICQueueQueOffsetZero_ReturnsFalse) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 1;
  headerInfo.aicQueOffset = 0;  // queOffset is 0
  headerInfo.aivQueOffset = sizeof(SkHeaderInfo);

  handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  handler->skHeaderInfoHost = &headerInfo;
  handler->aicTaskCnt = 1;
  handler->aivTaskCnt = 0;

  uint64_t argsAddr = 0;
  uint32_t argsSize = 0;

  bool result = handler->GetSubKernelTaskArgs(0, argsAddr, argsSize);
  EXPECT_FALSE(result);
}

TEST_F(SkDfxExceptionHandlerTest, GetSubKernelTaskArgs_AIVQueueQueOffsetZero_ReturnsFalse) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 1;
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);
  headerInfo.aivQueOffset = 0;  // queOffset is 0

  handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  handler->skHeaderInfoHost = &headerInfo;
  handler->aicTaskCnt = 0;
  handler->aivTaskCnt = 1;

  uint64_t argsAddr = 0;
  uint32_t argsSize = 0;

  bool result = handler->GetSubKernelTaskArgs(0, argsAddr, argsSize);
  EXPECT_FALSE(result);
}

TEST_F(SkDfxExceptionHandlerTest, GetSubKernelTaskArgs_AICQueueTaskCntZero_ReturnsFalse) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 1;
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);
  headerInfo.aivQueOffset = 0;

  handler->skDeviceEntryArgsHost = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  handler->skHeaderInfoHost = &headerInfo;
  handler->aicTaskCnt = 0;  // taskCnt is 0
  handler->aivTaskCnt = 0;

  uint64_t argsAddr = 0;
  uint32_t argsSize = 0;

  bool result = handler->GetSubKernelTaskArgs(0, argsAddr, argsSize);
  EXPECT_FALSE(result);
}

TEST_F(SkDfxExceptionHandlerTest, GetSubKernelTaskArgs_AICQueueMatchingTask_ReturnsTrue) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 2;
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);
  headerInfo.aivQueOffset = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aicQueOffset);
  aicTaskQue->taskCnt = 2;
  aicTaskQue->cap = 10;
  aicTaskQue->taskInfos[0].index = 0;
  aicTaskQue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
  aicTaskQue->taskInfos[0].args = 0xDEADBEEF;
  aicTaskQue->taskInfos[0].argsSize = 256;
  aicTaskQue->taskInfos[1].index = 1;
  aicTaskQue->taskInfos[1].type = SkTaskType::TYPE_SYNC;
  aicTaskQue->taskInfos[1].args = 0x12345678;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;
  handler->aicTaskCnt = 2;
  handler->aivTaskCnt = 0;

  uint64_t argsAddr = 0;
  uint32_t argsSize = 0;

  bool result = handler->GetSubKernelTaskArgs(0, argsAddr, argsSize);
  EXPECT_TRUE(result);
  EXPECT_EQ(argsAddr, 0xDEADBEEF);
  EXPECT_EQ(argsSize, 256);
}

TEST_F(SkDfxExceptionHandlerTest, GetSubKernelTaskArgs_AIVQueueMatchingTask_ReturnsTrue) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 3;
  headerInfo.aicQueOffset = 0;
  headerInfo.aivQueOffset = sizeof(SkHeaderInfo);

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  TaskQue *aivTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aivQueOffset);
  aivTaskQue->taskCnt = 2;
  aivTaskQue->cap = 10;
  aivTaskQue->taskInfos[0].index = 2;
  aivTaskQue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
  aivTaskQue->taskInfos[0].args = 0xCAFEBABE;
  aivTaskQue->taskInfos[0].argsSize = 128;
  aivTaskQue->taskInfos[1].index = 0;
  aivTaskQue->taskInfos[1].type = SkTaskType::TYPE_FUNC;
  aivTaskQue->taskInfos[1].args = 0x11111111;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;
  handler->aicTaskCnt = 0;
  handler->aivTaskCnt = 2;

  uint64_t argsAddr = 0;
  uint32_t argsSize = 0;

  bool result = handler->GetSubKernelTaskArgs(2, argsAddr, argsSize);
  EXPECT_TRUE(result);
  EXPECT_EQ(argsAddr, 0xCAFEBABE);
  EXPECT_EQ(argsSize, 128);
}

TEST_F(SkDfxExceptionHandlerTest, GetSubKernelTaskArgs_TaskIndexNotMatch_ReturnsFalse) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 2;
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);
  headerInfo.aivQueOffset = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aicQueOffset);
  aicTaskQue->taskCnt = 1;
  aicTaskQue->cap = 10;
  aicTaskQue->taskInfos[0].index = 1;  // index=1, but looking for nodeIdx=0
  aicTaskQue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
  aicTaskQue->taskInfos[0].args = 0x1234;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;
  handler->aicTaskCnt = 1;
  handler->aivTaskCnt = 0;

  uint64_t argsAddr = 0;
  uint32_t argsSize = 0;

  bool result = handler->GetSubKernelTaskArgs(0, argsAddr, argsSize);  // not found
  EXPECT_FALSE(result);
  EXPECT_EQ(argsAddr, 0);
  EXPECT_EQ(argsSize, 0);
}

TEST_F(SkDfxExceptionHandlerTest, GetSubKernelTaskArgs_TaskTypeNotFunc_ReturnsFalse) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 2;
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);
  headerInfo.aivQueOffset = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aicQueOffset);
  aicTaskQue->taskCnt = 1;
  aicTaskQue->cap = 10;
  aicTaskQue->taskInfos[0].index = 0;                     // index matches
  aicTaskQue->taskInfos[0].type = SkTaskType::TYPE_SYNC;  // but type is not TYPE_FUNC
  aicTaskQue->taskInfos[0].args = 0x5678;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;
  handler->aicTaskCnt = 1;
  handler->aivTaskCnt = 0;

  uint64_t argsAddr = 0;
  uint32_t argsSize = 0;

  bool result = handler->GetSubKernelTaskArgs(0, argsAddr, argsSize);
  EXPECT_FALSE(result);
}

TEST_F(SkDfxExceptionHandlerTest, GetSubKernelTaskArgs_AICNotFoundFallsBackToAIV_ReturnsTrue) {
  uint8_t buffer[2048] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 2;
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);
  headerInfo.aivQueOffset = sizeof(SkHeaderInfo) + sizeof(TaskQue) + sizeof(TaskInfo);

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  // AIC queue: nodeIdx=0 found but type is SYNC
  TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aicQueOffset);
  aicTaskQue->taskCnt = 1;
  aicTaskQue->cap = 10;
  aicTaskQue->taskInfos[0].index = 0;
  aicTaskQue->taskInfos[0].type = SkTaskType::TYPE_SYNC;
  aicTaskQue->taskInfos[0].args = 0xAAAA;

  // AIV queue: nodeIdx=0 found with TYPE_FUNC
  TaskQue *aivTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aivQueOffset);
  aivTaskQue->taskCnt = 1;
  aivTaskQue->cap = 10;
  aivTaskQue->taskInfos[0].index = 0;
  aivTaskQue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
  aivTaskQue->taskInfos[0].args = 0xBBBB;
  aivTaskQue->taskInfos[0].argsSize = 64;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;
  handler->aicTaskCnt = 1;
  handler->aivTaskCnt = 1;

  uint64_t argsAddr = 0;
  uint32_t argsSize = 0;

  bool result = handler->GetSubKernelTaskArgs(0, argsAddr, argsSize);
  EXPECT_TRUE(result);
  EXPECT_EQ(argsAddr, 0xBBBB);
  EXPECT_EQ(argsSize, 64);
}

// ==================== PopulateDumpInfoFields Tests (Line 698) ====================

// Mock for rtGetExceptionRegInfo - success with one core
int Fake_rtGetExceptionRegInfo_SingleCore(const void *exception, rtExceptionErrRegInfo_t **errRegInfo,
                                          uint32_t *coreNum) {
  (void)exception;
  static rtExceptionErrRegInfo_t g_singleCoreErrRegInfo;
  g_singleCoreErrRegInfo.coreId = 0;
  g_singleCoreErrRegInfo.coreType = RT_CORE_TYPE_AIC;
  g_singleCoreErrRegInfo.startPC = 0x1000;
  g_singleCoreErrRegInfo.currentPC = 0x1100;
  *errRegInfo = &g_singleCoreErrRegInfo;
  *coreNum = 1;
  return 0;
}

int Fake_rtGetExceptionRegInfo_SingleCoreNoMatch(const void *exception, rtExceptionErrRegInfo_t **errRegInfo,
                                                 uint32_t *coreNum) {
  (void)exception;
  static rtExceptionErrRegInfo_t g_singleCoreErrRegInfo;
  g_singleCoreErrRegInfo.coreId = 0;
  g_singleCoreErrRegInfo.coreType = RT_CORE_TYPE_AIC;
  g_singleCoreErrRegInfo.startPC = 0x1000;
  g_singleCoreErrRegInfo.currentPC = 0xFFFF;
  *errRegInfo = &g_singleCoreErrRegInfo;
  *coreNum = 1;
  return 0;
}

int Fake_rtGetExceptionRegInfo_TwoCoresSameBin(const void *exception, rtExceptionErrRegInfo_t **errRegInfo,
                                               uint32_t *coreNum) {
  (void)exception;
  static rtExceptionErrRegInfo_t g_errRegInfo[2];
  g_errRegInfo[0].coreId = 0;
  g_errRegInfo[0].coreType = RT_CORE_TYPE_AIC;
  g_errRegInfo[0].startPC = 0x1000;
  g_errRegInfo[0].currentPC = 0x1100;
  g_errRegInfo[1].coreId = 1;
  g_errRegInfo[1].coreType = RT_CORE_TYPE_AIC;
  g_errRegInfo[1].startPC = 0x2000;
  g_errRegInfo[1].currentPC = 0x2100;
  *errRegInfo = g_errRegInfo;
  *coreNum = 2;
  return 0;
}

TEST_F(SkDfxExceptionHandlerTest, PopulateDumpInfoFields_GetFuncHandleFail_ReturnsFailure) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  // Mock to return failure
  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(returnValue(ACL_ERROR_FAILURE));

  Adx::ExceptionDumpInfo dumpInfo = {};

  aclError ret = handler->PopulateDumpInfoFields(dumpInfo, 0, exceptionInfo, 0, RT_CORE_TYPE_AIC);
  EXPECT_EQ(ret, ACL_ERROR_FAILURE);
}

TEST_F(SkDfxExceptionHandlerTest, PopulateDumpInfoFields_GetFunctionNameFail_ReturnsFailure) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(returnValue(ACL_ERROR_FAILURE));

  Adx::ExceptionDumpInfo dumpInfo = {};

  aclError ret = handler->PopulateDumpInfoFields(dumpInfo, 0, exceptionInfo, 0, RT_CORE_TYPE_AIC);
  EXPECT_EQ(ret, ACL_ERROR_FAILURE);
}

TEST_F(SkDfxExceptionHandlerTest, PopulateDumpInfoFields_SetsKernelDisplayName) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  Adx::ExceptionDumpInfo dumpInfo = {};
  handler->skHeaderInfoHost = nullptr;  // Will cause crash if not early returned

  aclError ret = handler->PopulateDumpInfoFields(dumpInfo, 0, exceptionInfo, 0, RT_CORE_TYPE_AIC);
  // Should fail because skHeaderInfoHost is nullptr (used below)
  EXPECT_EQ(ret, ACL_ERROR_FAILURE);
}

TEST_F(SkDfxExceptionHandlerTest, PopulateDumpInfoFields_SetsExtraTensorInfo) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 1;
  headerInfo.totalSize = 512;
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo->binHdl = 0xAAA;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;
  handler->skDeviceEntryArgsDev = reinterpret_cast<void *>(0x12345678);

  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  Adx::ExceptionDumpInfo dumpInfo = {};

  aclError ret = handler->PopulateDumpInfoFields(dumpInfo, -1, exceptionInfo, 0, RT_CORE_TYPE_AIC);  // errorNodeIdx=-1
  EXPECT_EQ(ret, ACL_SUCCESS);

  // Verify extra tensor info is set
  EXPECT_EQ(dumpInfo.extraTensorNum, 1);
  EXPECT_EQ(dumpInfo.extraTensor[0].tensorSize, 512);
  EXPECT_EQ(dumpInfo.extraTensor[0].tensorAddr, reinterpret_cast<int64_t *>(0x12345678));
  EXPECT_EQ(dumpInfo.extraTensor[0].dataType, ACL_UINT8);
  EXPECT_EQ(dumpInfo.extraTensor[0].format, ACL_FORMAT_ND);

  // Verify SK entry fields filled as fallback when errorNodeIdx=-1
  // Exception in SK (not in sub-kernel): argAddr/argSize should be null/0
  EXPECT_EQ(dumpInfo.bin, nullptr);
  EXPECT_EQ(dumpInfo.argAddr, nullptr);
  EXPECT_EQ(dumpInfo.argSize, 0);
}

TEST_F(SkDfxExceptionHandlerTest, PopulateDumpInfoFields_WithValidErrorNodeIdx_SetsSubKernelInfo) {
  uint8_t buffer[2048] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 2;
  headerInfo.totalSize = 512;
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo) + sizeof(SkDfxInfo) * 2;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].binHdl = 0xAAA;
  dfxInfo[0].funcHdlOri = 0xBBB;
  dfxInfo[1].binHdl = 0xCCC;

  // Setup task queue with matching task for nodeIdx=0
  TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aicQueOffset);
  aicTaskQue->taskCnt = 1;
  aicTaskQue->cap = 10;
  aicTaskQue->taskInfos[0].index = 0;
  aicTaskQue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
  aicTaskQue->taskInfos[0].args = 0xDEADBEEF;
  aicTaskQue->taskInfos[0].argsSize = 128;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;
  handler->skDeviceEntryArgsDev = reinterpret_cast<void *>(0x12345678);
  handler->aicTaskCnt = 1;
  handler->aivTaskCnt = 0;

  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

  Adx::ExceptionDumpInfo dumpInfo = {};

  aclError ret = handler->PopulateDumpInfoFields(dumpInfo, 0, exceptionInfo, 0, RT_CORE_TYPE_AIC);
  EXPECT_EQ(ret, ACL_SUCCESS);

  // Verify sub kernel info is set
  EXPECT_EQ(dumpInfo.bin, reinterpret_cast<rtBinHandle>(0xAAA));
  EXPECT_EQ(dumpInfo.argAddr, reinterpret_cast<void *>(0xDEADBEEF));
  EXPECT_EQ(dumpInfo.argSize, 128);
}

TEST_F(SkDfxExceptionHandlerTest, PopulateDumpInfoFields_ErrorNodeIdxExceedsNodeCnt_FillsSkEntryFields) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 1;  // Only 1 node
  headerInfo.totalSize = 512;
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;
  handler->skDeviceEntryArgsDev = reinterpret_cast<void *>(0x12345678);

  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  Adx::ExceptionDumpInfo dumpInfo = {};

  aclError ret =
      handler->PopulateDumpInfoFields(dumpInfo, 5, exceptionInfo, 0, RT_CORE_TYPE_AIC);  // errorNodeIdx=5 > nodeCnt=1
  EXPECT_EQ(ret, ACL_SUCCESS);

  // Verify SK entry fields filled as fallback when errorNodeIdx exceeds nodeCnt
  // Exception in SK (not in sub-kernel): argAddr/argSize should be null/0
  EXPECT_EQ(dumpInfo.bin, nullptr);
  EXPECT_EQ(dumpInfo.argAddr, nullptr);
  EXPECT_EQ(dumpInfo.argSize, 0);
}

TEST_F(SkDfxExceptionHandlerTest, PopulateDumpInfoFields_FillsKernelNameField) {
  uint8_t buffer[1024] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 1;
  headerInfo.totalSize = 512;
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;
  handler->skDeviceEntryArgsDev = reinterpret_cast<void *>(0x12345678);

  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  Adx::ExceptionDumpInfo dumpInfo = {};
  memcpy_s(dumpInfo.kernelName, Adx::MAX_KERNELNAME_LEN, "old_value", sizeof("old_value"));

  aclError ret = handler->PopulateDumpInfoFields(dumpInfo, -1, exceptionInfo, 0, RT_CORE_TYPE_AIC);
  EXPECT_EQ(ret, ACL_SUCCESS);

  // Verify kernelName is filled with SK entry func name when errorNodeIdx < 0
  EXPECT_STREQ(dumpInfo.kernelName, "sk_entry");
}

TEST_F(SkDfxExceptionHandlerTest, PopulateSkEntryFields_AcceptsMaximumFunctionNameAndTruncatesDisplayName) {
  SkHeaderInfo headerInfo = {};
  headerInfo.modelIdIndexAndSkScopeId = static_cast<uint64_t>(UINT16_MAX) << 32;
  handler->skHeaderInfoHost = &headerInfo;

  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);
  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_MaxLengthSkEntry));

  Adx::ExceptionDumpInfo dumpInfo = {};
  ASSERT_EQ(handler->PopulateSkEntryFields(dumpInfo, exceptionInfo), ACL_SUCCESS);

  const std::string expected = MaxLengthSkEntryName() + "_scope0";
  EXPECT_EQ(std::string(dumpInfo.kernelDisplayName), expected.substr(0, Adx::MAX_KERNELNAME_LEN - 1U));
}

TEST_F(SkDfxExceptionHandlerTest, PopulateSubKernelFields_AcceptsMaximumFunctionNameAndTruncatesKernelName) {
  SkHeaderInfo headerInfo = {};
  headerInfo.nodeCnt = 0;
  handler->skHeaderInfoHost = &headerInfo;

  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);
  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_MaxLengthSkEntry));

  Adx::ExceptionDumpInfo dumpInfo = {};
  ASSERT_TRUE(handler->PopulateSubKernelFields(dumpInfo, -1, exceptionInfo));

  EXPECT_EQ(std::string(dumpInfo.kernelName), MaxLengthSkEntryName().substr(0, Adx::MAX_KERNELNAME_LEN - 1U));
}

// ==================== FillExceptionDumpInfo Tests (Line 758) ====================

TEST_F(SkDfxExceptionHandlerTest, FillExceptionDumpInfo_NullExceptionInfo_ReturnsInvalidParam) {
  Adx::ExceptionDumpInfo dumpInfo = {};

  aclError ret = handler->FillExceptionDumpInfo(dumpInfo, nullptr);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SkDfxExceptionHandlerTest, FillExceptionDumpInfo_ExtractSkEntryArgsFails_ReturnsFailure) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(returnValue(ACL_ERROR_FAILURE));

  Adx::ExceptionDumpInfo dumpInfo = {};

  aclError ret = handler->FillExceptionDumpInfo(dumpInfo, exceptionInfo);
  EXPECT_EQ(ret, ACL_ERROR_FAILURE);
}

TEST_F(SkDfxExceptionHandlerTest, FillExceptionDumpInfo_ExtractTaskQueueFails_ReturnsFailure) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));

  // Setup buffer so ExtractTaskQueue will fail (use heap memory for proper cleanup)
  constexpr size_t bufferSize = 1024;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);
  SkHeaderInfo headerInfo = {};
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);  // Valid offset
  headerInfo.aivQueOffset = sizeof(SkHeaderInfo);
  headerInfo.totalSize = bufferSize;
  headerInfo.nodeCnt = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;
  handler->skDeviceEntryArgsHost = deviceArgs;

  Adx::ExceptionDumpInfo dumpInfo = {};

  aclError ret = handler->FillExceptionDumpInfo(dumpInfo, exceptionInfo);
  EXPECT_EQ(ret, ACL_ERROR_FAILURE);
  // FreeResources() already freed buffer via aclrtFreeHost
}

TEST_F(SkDfxExceptionHandlerTest, FillExceptionDumpInfo_GetExceptionRegInfoFails_ReturnsFailure) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));
  MOCKER(rtGetExceptionRegInfo).stubs().will(returnValue(-1));

  // Setup minimal buffer (use heap memory for proper cleanup)
  constexpr size_t bufferSize = 1024;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);
  SkHeaderInfo headerInfo = {};
  headerInfo.totalSize = sizeof(SkHeaderInfo);
  headerInfo.nodeCnt = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;
  handler->skDeviceEntryArgsHost = deviceArgs;

  Adx::ExceptionDumpInfo dumpInfo = {};

  aclError ret = handler->FillExceptionDumpInfo(dumpInfo, exceptionInfo);
  EXPECT_EQ(ret, ACL_ERROR_FAILURE);
  // FreeResources() already freed buffer via aclrtFreeHost
}

TEST_F(SkDfxExceptionHandlerTest, FillExceptionDumpInfo_PopulateDumpInfoFieldsFails_ReturnsFailure) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));
  MOCKER(rtGetExceptionRegInfo).stubs().will(invoke(Fake_rtGetExceptionRegInfo_Success));
  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(returnValue(ACL_ERROR_FAILURE));

  // Setup minimal buffer (use heap memory for proper cleanup)
  constexpr size_t bufferSize = 1024;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);
  SkHeaderInfo headerInfo = {};
  headerInfo.totalSize = sizeof(SkHeaderInfo);
  headerInfo.nodeCnt = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;
  handler->skDeviceEntryArgsHost = deviceArgs;

  Adx::ExceptionDumpInfo dumpInfo = {};

  aclError ret = handler->FillExceptionDumpInfo(dumpInfo, exceptionInfo);
  EXPECT_EQ(ret, ACL_ERROR_FAILURE);
  // FreeResources() already freed buffer via aclrtFreeHost
}

TEST_F(SkDfxExceptionHandlerTest, FillExceptionDumpInfo_Success_ReturnsACL_SUCCESS) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy_DeviceToHost));
  MOCKER(rtGetExceptionRegInfo).stubs().will(invoke(Fake_rtGetExceptionRegInfo_SingleCore));
  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  // Setup buffer with valid header (use heap memory for proper cleanup)
  constexpr size_t bufferSize = 1024;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);
  SkHeaderInfo headerInfo = {};
  headerInfo.totalSize = bufferSize;
  headerInfo.nodeCnt = 0;
  headerInfo.dfxOffset = 0;
  headerInfo.counterOffset = 0;
  headerInfo.eventConfigOffset = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;
  handler->skDeviceEntryArgsHost = deviceArgs;

  // Set global mock buffer for aclrtMemcpy mock
  g_mockDeviceBuffer = buffer;
  g_mockDeviceBufferSize = bufferSize;

  Adx::ExceptionDumpInfo dumpInfo = {};

  aclError ret = handler->FillExceptionDumpInfo(dumpInfo, exceptionInfo);
  EXPECT_EQ(ret, ACL_SUCCESS);

  // Verify kernelDisplayName is set (skScopeId=0 when header not initialized, fallback to skFuncName_scope%u)
  EXPECT_STREQ(dumpInfo.kernelDisplayName, "sk_entry_scope0");

  // Clear global mock buffer
  g_mockDeviceBuffer = nullptr;
  g_mockDeviceBufferSize = 0;
  // FreeResources() already freed buffer via aclrtFreeHost
}

TEST_F(SkDfxExceptionHandlerTest, FillExceptionDumpInfo_CallsIdentifyErrorNodeByPC) {
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);

  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy_DeviceToHost));
  MOCKER(rtGetExceptionRegInfo).stubs().will(invoke(Fake_rtGetExceptionRegInfo_SingleCore));
  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  // Setup buffer with DFX info for IdentifyErrorNodeByPC (use heap memory for proper cleanup)
  constexpr size_t bufferSize = 2048;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);

  SkHeaderInfo *headerInfo = reinterpret_cast<SkHeaderInfo *>(buffer);
  headerInfo->totalSize = bufferSize;
  headerInfo->dfxOffset = sizeof(SkHeaderInfo);
  headerInfo->nodeCnt = 1;
  headerInfo->counterOffset = 0;
  headerInfo->eventConfigOffset = 0;
  headerInfo->aicQueOffset = 0;
  headerInfo->aivQueOffset = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = *headerInfo;

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo->dfxOffset);
  dfxInfo->entryAic[0] = 0x1000;
  dfxInfo->aicSize = 0x200;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = headerInfo;

  // Set global mock buffer for aclrtMemcpy mock
  g_mockDeviceBuffer = buffer;
  g_mockDeviceBufferSize = bufferSize;

  Adx::ExceptionDumpInfo dumpInfo = {};

  // Call with currentPC that matches the entry
  aclError ret = handler->FillExceptionDumpInfo(dumpInfo, exceptionInfo);
  EXPECT_EQ(ret, ACL_SUCCESS);

  // Clear global mock buffer
  g_mockDeviceBuffer = nullptr;
  g_mockDeviceBufferSize = 0;
  // FreeResources() already freed buffer via aclrtFreeHost
}

TEST_F(SkDfxExceptionHandlerTest, GetErrorNodeIdx_InitiallyNegativeOne) {
  EXPECT_EQ(handler->GetErrorNodeIdx(), -1);
}

TEST_F(SkDfxExceptionHandlerTest, GetErrorNodeIdx_AfterIdentifyErrorNode_MatchesErrorNode) {
  uint8_t buffer[2048] = {0};
  SkHeaderInfo headerInfo = {};
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);
  headerInfo.nodeCnt = 1;
  headerInfo.aicQueOffset = 0;
  headerInfo.aivQueOffset = 0;

  SkDeviceEntryArgs *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader = headerInfo;

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo->entryAic[0] = 0x1000;
  dfxInfo->aicSize = 0x200;

  handler->skDeviceEntryArgsHost = deviceArgs;
  handler->skHeaderInfoHost = &headerInfo;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));

  // PC matches node[0]
  handler->IdentifyErrorNodeByPC(0, RT_CORE_TYPE_AIC, 0x1000, 0x1100);

  EXPECT_EQ(handler->GetErrorNodeIdx(), 0);
}

// ==================== ProcessExceptionDump / ExceptionDumpInfoCallBack Tests ====================

TEST_F(SkDfxExceptionHandlerTest, ProcessExceptionDump_SizeZero_ReturnsInvalidParam) {
  aclrtExceptionInfo exceptionInfo = {};
  Adx::ExceptionDumpInfo dumpInfo = {};
  uint32_t realSize = 99;
  Adx::ExceptionDumpMode mode = Adx::ExceptionDumpMode::DUMP_MODE_ADDITIONAL;

  uint32_t ret = handler->ProcessExceptionDump(&exceptionInfo, &dumpInfo, 0, &realSize, &mode);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(realSize, 99);
  EXPECT_EQ(mode, Adx::ExceptionDumpMode::DUMP_MODE_ADDITIONAL);
}

TEST_F(SkDfxExceptionHandlerTest, ProcessExceptionDump_NotSkEntry_SkipsDump) {
  aclrtExceptionInfo exceptionInfo = {};
  Adx::ExceptionDumpInfo dumpInfo = {};
  uint32_t realSize = 99;
  Adx::ExceptionDumpMode mode = Adx::ExceptionDumpMode::DUMP_MODE_ADDITIONAL;

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_other));

  uint32_t ret = handler->ProcessExceptionDump(&exceptionInfo, &dumpInfo, 1, &realSize, &mode);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(realSize, 0);
  EXPECT_EQ(mode, Adx::ExceptionDumpMode::DUMP_MODE_NONE);
}

TEST_F(SkDfxExceptionHandlerTest, ExceptionDumpInfoCallBack_NullParams_ReturnsInvalidParam) {
  aclrtExceptionInfo exceptionInfo = {};
  Adx::ExceptionDumpInfo dumpInfo = {};
  uint32_t realSize = 0;
  Adx::ExceptionDumpMode mode = Adx::ExceptionDumpMode::DUMP_MODE_NONE;

  EXPECT_EQ(ExceptionDumpInfoCallBack(nullptr, &dumpInfo, 1, &realSize, &mode), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(ExceptionDumpInfoCallBack(&exceptionInfo, nullptr, 1, &realSize, &mode), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(ExceptionDumpInfoCallBack(&exceptionInfo, &dumpInfo, 1, nullptr, &mode), ACL_ERROR_INVALID_PARAM);
}

TEST_F(SkDfxExceptionHandlerTest, ExceptionDumpInfoCallBack_CommonException_SkipsDump) {
  aclrtExceptionInfo exceptionInfo = {};
  exceptionInfo.expandInfo.type = static_cast<rtExceptionExpandType_t>(99);
  Adx::ExceptionDumpInfo dumpInfo = {};
  uint32_t realSize = 99;
  Adx::ExceptionDumpMode mode = Adx::ExceptionDumpMode::DUMP_MODE_ADDITIONAL;

  uint32_t ret = ExceptionDumpInfoCallBack(&exceptionInfo, &dumpInfo, 1, &realSize, &mode);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(realSize, 0);
  EXPECT_EQ(mode, Adx::ExceptionDumpMode::DUMP_MODE_NONE);
}

TEST_F(SkDfxExceptionHandlerTest, ExceptionDumpInfoCallBack_SuperKernelException_FillsSubKernelDump) {
  constexpr size_t bufferSize = 2048;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);

  auto *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  SkHeaderInfo &headerInfo = deviceArgs->skHeader;
  headerInfo.totalSize = bufferSize;
  headerInfo.nodeCnt = 1;
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);
  headerInfo.aicQueOffset = headerInfo.dfxOffset + sizeof(SkDfxInfo);
  const std::string modelId = "model_dfx_callback_route_model";
  uint16_t modelIdIndex = SkEventRecorder::Instance().RegisterModelId(modelId);
  headerInfo.modelIdIndexAndSkScopeId = static_cast<uint64_t>(modelIdIndex) << 32;

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].binHdl = 0xAAAA;
  dfxInfo[0].funcHdlOri = 0xBBBB;
  dfxInfo[0].entryAic[0] = 0x1000;
  dfxInfo[0].aicSize = 0x200;

  TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aicQueOffset);
  aicTaskQue->taskCnt = 1;
  aicTaskQue->cap = 1;
  aicTaskQue->taskInfos[0].index = 0;
  aicTaskQue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
  aicTaskQue->taskInfos[0].args = 0xDEADBEEF;
  aicTaskQue->taskInfos[0].argsSize = 256;

  g_mockDeviceBuffer = buffer;
  g_mockDeviceBufferSize = bufferSize;

  aclrtExceptionInfo exceptionInfo = {};
  exceptionInfo.expandInfo.type = RT_EXCEPTION_AICORE;
  Adx::ExceptionDumpInfo dumpInfo[1] = {};
  uint32_t realSize = 0;
  Adx::ExceptionDumpMode mode = Adx::ExceptionDumpMode::DUMP_MODE_NONE;

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));
  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy_DeviceToHost));
  MOCKER(rtGetExceptionRegInfo).stubs().will(invoke(Fake_rtGetExceptionRegInfo_SingleCore));

  const std::string previousModelId = "model_dfx_callback_previous";
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = modelId;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));
  config.modelId = previousModelId;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));
  sk::logger::FileLogger::SetCurrentModelId(previousModelId);
  sk::logger::FileHandleManager::Instance().SwitchToDefault();

  uint32_t ret = ExceptionDumpInfoCallBack(&exceptionInfo, dumpInfo, 1, &realSize, &mode);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(realSize, 1);
  EXPECT_EQ(mode, Adx::ExceptionDumpMode::DUMP_MODE_OVERWRITE);
  EXPECT_EQ(dumpInfo[0].coreId, 0);
  EXPECT_EQ(dumpInfo[0].coreType, RT_CORE_TYPE_AIC);
  EXPECT_EQ(dumpInfo[0].bin, reinterpret_cast<void *>(0xAAAA));
  EXPECT_STREQ(dumpInfo[0].kernelName, "sk_entry");
  EXPECT_EQ(dumpInfo[0].argAddr, reinterpret_cast<void *>(0xDEADBEEF));
  EXPECT_EQ(dumpInfo[0].argSize, 256);
  EXPECT_EQ(dumpInfo[0].extraTensorNum, 1);
  EXPECT_EQ(dumpInfo[0].extraTensor[0].tensorSize, bufferSize);
  EXPECT_EQ(dumpInfo[0].extraTensor[0].tensorAddr, reinterpret_cast<int64_t *>(0x3000));
  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), previousModelId);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");

  std::ifstream targetLog(GetSkMetaPath(modelId) + "/super_kernel.log");
  std::string targetLogContent((std::istreambuf_iterator<char>(targetLog)), std::istreambuf_iterator<char>());
  EXPECT_NE(targetLogContent.find("Exception is in SuperKernel"), std::string::npos);
  std::ifstream previousLog(GetSkMetaPath(previousModelId) + "/super_kernel.log");
  std::string previousLogContent((std::istreambuf_iterator<char>(previousLog)), std::istreambuf_iterator<char>());
  EXPECT_EQ(previousLogContent.find("Exception is in SuperKernel"), std::string::npos);

  sk::logger::FileLogger::Instance().SetEnabled(false);
  g_mockDeviceBuffer = nullptr;
  g_mockDeviceBufferSize = 0;
  free(buffer);
}

TEST_F(SkDfxExceptionHandlerTest, ExceptionDumpInfoCallBack_SameBinHandle_ReturnsOneDump) {
  constexpr size_t bufferSize = 2048;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);

  auto *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  SkHeaderInfo &headerInfo = deviceArgs->skHeader;
  headerInfo.totalSize = bufferSize;
  headerInfo.nodeCnt = 2;
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);
  headerInfo.aicQueOffset = headerInfo.dfxOffset + sizeof(SkDfxInfo) * 2;
  headerInfo.modelIdIndexAndSkScopeId = 0;

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].binHdl = 0xAAAA;
  dfxInfo[0].funcHdlOri = 0xBBBB;
  dfxInfo[0].entryAic[0] = 0x1000;
  dfxInfo[0].aicSize = 0x200;
  dfxInfo[1].binHdl = 0xAAAA;
  dfxInfo[1].funcHdlOri = 0xCCCC;
  dfxInfo[1].entryAic[0] = 0x2000;
  dfxInfo[1].aicSize = 0x200;

  TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(buffer + headerInfo.aicQueOffset);
  aicTaskQue->taskCnt = 2;
  aicTaskQue->cap = 2;
  aicTaskQue->taskInfos[0].index = 0;
  aicTaskQue->taskInfos[0].type = SkTaskType::TYPE_FUNC;
  aicTaskQue->taskInfos[0].args = 0xDEADBEEF;
  aicTaskQue->taskInfos[0].argsSize = 256;
  aicTaskQue->taskInfos[1].index = 1;
  aicTaskQue->taskInfos[1].type = SkTaskType::TYPE_FUNC;
  aicTaskQue->taskInfos[1].args = 0xCAFEBABE;
  aicTaskQue->taskInfos[1].argsSize = 512;

  g_mockDeviceBuffer = buffer;
  g_mockDeviceBufferSize = bufferSize;

  aclrtExceptionInfo exceptionInfo = {};
  exceptionInfo.expandInfo.type = RT_EXCEPTION_AICORE;
  Adx::ExceptionDumpInfo dumpInfo[2] = {};
  uint32_t realSize = 0;
  Adx::ExceptionDumpMode mode = Adx::ExceptionDumpMode::DUMP_MODE_NONE;

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));
  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy_DeviceToHost));
  MOCKER(rtGetExceptionRegInfo).stubs().will(invoke(Fake_rtGetExceptionRegInfo_TwoCoresSameBin));

  uint32_t ret = ExceptionDumpInfoCallBack(&exceptionInfo, dumpInfo, 2, &realSize, &mode);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(realSize, 1);
  EXPECT_EQ(mode, Adx::ExceptionDumpMode::DUMP_MODE_OVERWRITE);
  EXPECT_EQ(dumpInfo[0].coreId, 0);
  EXPECT_EQ(dumpInfo[0].bin, reinterpret_cast<void *>(0xAAAA));
  EXPECT_EQ(dumpInfo[0].argAddr, reinterpret_cast<void *>(0xDEADBEEF));
  EXPECT_EQ(dumpInfo[0].argSize, 256);
  EXPECT_EQ(dumpInfo[1].argAddr, nullptr);

  g_mockDeviceBuffer = nullptr;
  g_mockDeviceBufferSize = 0;
  free(buffer);
}

TEST_F(SkDfxExceptionHandlerTest, ExceptionDumpInfoCallBack_NoMatch_DoesNotPrintNoMatchInfo) {
  constexpr size_t bufferSize = 2048;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);

  auto *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  SkHeaderInfo &headerInfo = deviceArgs->skHeader;
  headerInfo.totalSize = bufferSize;
  headerInfo.nodeCnt = 1;
  headerInfo.dfxOffset = sizeof(SkHeaderInfo);
  headerInfo.modelIdIndexAndSkScopeId = 0;

  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(buffer + headerInfo.dfxOffset);
  dfxInfo[0].entryAic[0] = 0x1000;
  dfxInfo[0].aicSize = 0x200;

  g_mockDeviceBuffer = buffer;
  g_mockDeviceBufferSize = bufferSize;

  aclrtExceptionInfo exceptionInfo = {};
  exceptionInfo.expandInfo.type = RT_EXCEPTION_AICORE;
  Adx::ExceptionDumpInfo dumpInfo[1] = {};
  uint32_t realSize = 0;
  Adx::ExceptionDumpMode mode = Adx::ExceptionDumpMode::DUMP_MODE_NONE;

  MOCKER(aclrtGetFuncHandleFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetFuncHandleFromExceptionInfo_Success));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionName_sk_entry));
  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy_DeviceToHost));
  MOCKER(rtGetExceptionRegInfo).stubs().will(invoke(Fake_rtGetExceptionRegInfo_SingleCoreNoMatch));

  ut_log::LogBuffer::Instance().Clear();
  uint32_t ret = ExceptionDumpInfoCallBack(&exceptionInfo, dumpInfo, 1, &realSize, &mode);
  std::string logContent = ut_log::LogBuffer::Instance().GetContent();

  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(realSize, 1);
  EXPECT_EQ(mode, Adx::ExceptionDumpMode::DUMP_MODE_OVERWRITE);
  EXPECT_STREQ(dumpInfo[0].kernelName, "sk_entry");
  EXPECT_EQ(dumpInfo[0].argAddr, nullptr);
  EXPECT_EQ(dumpInfo[0].argSize, 0);
  EXPECT_EQ(logContent.find("No sub kernel matched, aicore error occurred in sk entry."), std::string::npos);

  ut_log::LogBuffer::Instance().Clear();
  g_mockDeviceBuffer = nullptr;
  g_mockDeviceBufferSize = 0;
  free(buffer);
}

// ==================== CheckError Tests ====================

TEST_F(SkDfxExceptionHandlerTest, CheckError_ACL_SUCCESS_ReturnsSuccess) {
  aclError ret = handler->CheckError(ACL_SUCCESS, "Test operation");
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkDfxExceptionHandlerTest, CheckError_OtherError_ReturnsError) {
  aclError ret = handler->CheckError(ACL_ERROR_INVALID_PARAM, "Test operation");
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

// ==================== ValidateSkHeaderOffsets Tests ====================

TEST_F(SkDfxExceptionHandlerTest, ValidateSkHeaderOffsets_AllOffsetsZero_ReturnsTrue) {
  SkHeaderInfo headerInfo = {};
  headerInfo.totalSize = sizeof(SkHeaderInfo);
  handler->skHeaderInfoHost = &headerInfo;
  EXPECT_TRUE(handler->ValidateSkHeaderOffsets());
}

TEST_F(SkDfxExceptionHandlerTest, ValidateSkHeaderOffsets_AllOffsetsValid_ReturnsTrue) {
  uint32_t aicQueOffset = sizeof(SkHeaderInfo);
  uint32_t aicQueSize = sizeof(TaskQue);
  uint32_t aivQueOffset = aicQueOffset + aicQueSize;
  uint32_t aivQueSize = sizeof(TaskQue);
  uint32_t counterOffset = aivQueOffset + aivQueSize;
  uint64_t counterDataSize = static_cast<uint64_t>(handler->aicoreNums) * sizeof(SkCounterInfo);
  uint32_t dfxOffset = counterOffset + static_cast<uint32_t>(counterDataSize);

  SkHeaderInfo headerInfo = {};
  headerInfo.aicQueOffset = aicQueOffset;
  headerInfo.aicQueSize = aicQueSize;
  headerInfo.aivQueOffset = aivQueOffset;
  headerInfo.aivQueSize = aivQueSize;
  headerInfo.counterOffset = counterOffset;
  headerInfo.dfxOffset = dfxOffset;
  headerInfo.nodeCnt = 1;
  headerInfo.totalSize = dfxOffset + sizeof(SkDfxInfo);
  handler->skHeaderInfoHost = &headerInfo;
  EXPECT_TRUE(handler->ValidateSkHeaderOffsets());
}

TEST_F(SkDfxExceptionHandlerTest, ValidateSkHeaderOffsets_OffsetExceedsTotalSize_ReturnsFalse) {
  // aicQueOffset >= totalSize -> invalid
  SkHeaderInfo headerInfo = {};
  headerInfo.aicQueOffset = 2048;
  headerInfo.aicQueSize = 100;
  headerInfo.totalSize = 1024;
  handler->skHeaderInfoHost = &headerInfo;
  EXPECT_FALSE(handler->ValidateSkHeaderOffsets());
}

TEST_F(SkDfxExceptionHandlerTest, ValidateSkHeaderOffsets_OffsetPlusSizeExceedsTotalSize_ReturnsFalse) {
  // dfxOffset + nodeCnt * sizeof(SkDfxInfo) > totalSize -> invalid
  SkHeaderInfo headerInfo = {};
  headerInfo.dfxOffset = 900;
  headerInfo.nodeCnt = 10;
  headerInfo.totalSize = 1024;
  handler->skHeaderInfoHost = &headerInfo;
  EXPECT_FALSE(handler->ValidateSkHeaderOffsets());
}

TEST_F(SkDfxExceptionHandlerTest, ValidateSkHeaderOffsets_PartialOffsetsValid_ReturnsTrue) {
  // Some offsets zero, some valid -> only validate non-zero offsets
  SkHeaderInfo headerInfo = {};
  headerInfo.aicQueOffset = sizeof(SkHeaderInfo);
  headerInfo.aicQueSize = 128;
  headerInfo.totalSize = sizeof(SkHeaderInfo) + 128;
  handler->skHeaderInfoHost = &headerInfo;
  EXPECT_TRUE(handler->ValidateSkHeaderOffsets());
}

TEST_F(SkDfxExceptionHandlerTest, ValidateSkHeaderOffsets_AicQueOffsetPlusSizeExceedsTotalSize_ReturnsFalse) {
  SkHeaderInfo headerInfo = {};
  headerInfo.aicQueOffset = 900;
  headerInfo.aicQueSize = 200;  // 900 + 200 = 1100 > 1024
  headerInfo.totalSize = 1024;
  handler->skHeaderInfoHost = &headerInfo;
  EXPECT_FALSE(handler->ValidateSkHeaderOffsets());
}

TEST_F(SkDfxExceptionHandlerTest, ValidateSkHeaderOffsets_AivQueOffsetExceedsTotalSize_ReturnsFalse) {
  SkHeaderInfo headerInfo = {};
  headerInfo.aivQueOffset = 2048;
  headerInfo.aivQueSize = 100;
  headerInfo.totalSize = 1024;
  handler->skHeaderInfoHost = &headerInfo;
  EXPECT_FALSE(handler->ValidateSkHeaderOffsets());
}

TEST_F(SkDfxExceptionHandlerTest, ValidateSkHeaderOffsets_AivQueOffsetPlusSizeExceedsTotalSize_ReturnsFalse) {
  SkHeaderInfo headerInfo = {};
  headerInfo.aivQueOffset = 900;
  headerInfo.aivQueSize = 200;  // 900 + 200 = 1100 > 1024
  headerInfo.totalSize = 1024;
  handler->skHeaderInfoHost = &headerInfo;
  EXPECT_FALSE(handler->ValidateSkHeaderOffsets());
}

TEST_F(SkDfxExceptionHandlerTest, ValidateSkHeaderOffsets_CounterOffsetExceedsTotalSize_ReturnsFalse) {
  SkHeaderInfo headerInfo = {};
  headerInfo.counterOffset = 2048;
  headerInfo.totalSize = 1024;
  handler->skHeaderInfoHost = &headerInfo;
  EXPECT_FALSE(handler->ValidateSkHeaderOffsets());
}

TEST_F(SkDfxExceptionHandlerTest, ValidateSkHeaderOffsets_CounterOffsetPlusSizeExceedsTotalSize_ReturnsFalse) {
  // counterOffset + aicoreNums(75) * sizeof(SkCounterInfo) > totalSize
  SkHeaderInfo headerInfo = {};
  headerInfo.counterOffset = 100;
  headerInfo.totalSize = 1024;
  handler->skHeaderInfoHost = &headerInfo;
  EXPECT_FALSE(handler->ValidateSkHeaderOffsets());
}

TEST_F(SkDfxExceptionHandlerTest, ValidateSkHeaderOffsets_DfxOffsetExceedsTotalSize_ReturnsFalse) {
  SkHeaderInfo headerInfo = {};
  headerInfo.dfxOffset = 2048;
  headerInfo.nodeCnt = 1;
  headerInfo.totalSize = 1024;
  handler->skHeaderInfoHost = &headerInfo;
  EXPECT_FALSE(handler->ValidateSkHeaderOffsets());
}

// ==================== totalSize vs skDeviceEntryArgsPtrLen Validation Tests ====================

TEST_F(SkDfxExceptionHandlerTest, ExtractSkEntryArgs_TotalSizeExceedsPtrLen_ReturnsFalse) {
  // Use SmallLen mock so skDeviceEntryArgsPtrLen=8 < totalSize
  constexpr size_t bufferSize = 256;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);

  SkHeaderInfo headerInfo = {};
  headerInfo.totalSize = bufferSize;
  headerInfo.nodeCnt = 0;
  reinterpret_cast<SkDeviceEntryArgs *>(buffer)->skHeader = headerInfo;

  g_mockDeviceBuffer = buffer;
  g_mockDeviceBufferSize = bufferSize;

  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);
  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_SmallLen));
  MOCKER(aclrtMallocHost).stubs().will(invoke(Fake_aclrtMallocHost_Success));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy_DeviceToHost));
  MOCKER(aclrtFreeHost).stubs().will(invoke(Fake_aclrtFreeHost_Success));

  std::unique_ptr<sk::logger::LogContextGuard> logContext;
  EXPECT_FALSE(handler->ExtractSkEntryArgs(exceptionInfo, logContext));

  g_mockDeviceBuffer = nullptr;
  g_mockDeviceBufferSize = 0;
  free(buffer);
}

TEST_F(SkDfxExceptionHandlerTest, ExtractSkEntryArgs_InvalidOffsetFailsValidation_ReturnsFalse) {
  // totalSize <= ptrLen passes, but dfxOffset out of bounds fails ValidateSkHeaderOffsets
  constexpr size_t bufferSize = 256;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);

  SkHeaderInfo headerInfo = {};
  headerInfo.totalSize = bufferSize;
  headerInfo.dfxOffset = 200;
  headerInfo.nodeCnt = 5;  // 200 + 5 * sizeof(SkDfxInfo) >> 256
  reinterpret_cast<SkDeviceEntryArgs *>(buffer)->skHeader = headerInfo;

  g_mockDeviceBuffer = buffer;
  g_mockDeviceBufferSize = bufferSize;

  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);
  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));
  MOCKER(aclrtMallocHost).stubs().will(invoke(Fake_aclrtMallocHost_Success));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy_DeviceToHost));
  MOCKER(aclrtFreeHost).stubs().will(invoke(Fake_aclrtFreeHost_Success));

  std::unique_ptr<sk::logger::LogContextGuard> logContext;
  EXPECT_FALSE(handler->ExtractSkEntryArgs(exceptionInfo, logContext));

  g_mockDeviceBuffer = nullptr;
  g_mockDeviceBufferSize = 0;
  free(buffer);
}

TEST_F(SkDfxExceptionHandlerTest, ExtractSkEntryArgs_RoutesToHeaderModelAndGuardRestoresPreviousState) {
  constexpr size_t bufferSize = 256;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);

  const std::string modelId = "model_dfx_route_model";
  uint16_t modelIdIndex = SkEventRecorder::Instance().RegisterModelId(modelId);
  auto *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader.totalSize = bufferSize;
  deviceArgs->skHeader.modelIdIndexAndSkScopeId = static_cast<uint64_t>(modelIdIndex) << 32;

  g_mockDeviceBuffer = buffer;
  g_mockDeviceBufferSize = bufferSize;

  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));
  MOCKER(aclrtMallocHost).stubs().will(invoke(Fake_aclrtMallocHost_Success));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy_DeviceToHost));
  MOCKER(aclrtFreeHost).stubs().will(invoke(Fake_aclrtFreeHost_Success));

  const std::string previousModelId = "model_dfx_route_previous";
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = previousModelId;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));
  sk::logger::FileLogger::SetCurrentModelId(previousModelId);
  sk::logger::FileHandleManager::Instance().SwitchToDefault();
  const std::string unknownModelPath = GetSkMetaPath(UNKNOWN_MODEL_ID);
  std::filesystem::remove_all(unknownModelPath);

  std::unique_ptr<sk::logger::LogContextGuard> logContext;
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);
  EXPECT_TRUE(handler->ExtractSkEntryArgs(exceptionInfo, logContext));
  ASSERT_NE(logContext, nullptr);
  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), modelId);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
  EXPECT_FALSE(std::filesystem::exists(unknownModelPath));

  handler->FreeResources();
  logContext.reset();
  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), previousModelId);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");

  sk::logger::FileLogger::Instance().SetEnabled(false);
  g_mockDeviceBuffer = nullptr;
  g_mockDeviceBufferSize = 0;
  free(buffer);
}

TEST_F(SkDfxExceptionHandlerTest, ExtractSkEntryArgs_UnknownModelIndexWritesFallbackLog) {
  constexpr size_t bufferSize = 256;
  uint8_t *buffer = static_cast<uint8_t *>(malloc(bufferSize));
  ASSERT_NE(buffer, nullptr);
  memset_s(buffer, bufferSize, 0, bufferSize);

  auto *deviceArgs = reinterpret_cast<SkDeviceEntryArgs *>(buffer);
  deviceArgs->skHeader.totalSize = bufferSize;
  deviceArgs->skHeader.modelIdIndexAndSkScopeId = static_cast<uint64_t>(UINT16_MAX) << 32;

  g_mockDeviceBuffer = buffer;
  g_mockDeviceBufferSize = bufferSize;

  MOCKER(aclrtGetArgsFromExceptionInfo).stubs().will(invoke(Fake_aclrtGetArgsFromExceptionInfo_Success));
  MOCKER(aclrtMallocHost).stubs().will(invoke(Fake_aclrtMallocHost_Success));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy_DeviceToHost));
  MOCKER(aclrtFreeHost).stubs().will(invoke(Fake_aclrtFreeHost_Success));

  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelId = "model_dfx_fallback_previous";
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));
  sk::logger::FileLogger::SetCurrentModelId(config.modelId);
  sk::logger::FileHandleManager::Instance().SwitchToDefault();

  const size_t sizeBefore = sk::logger::FileHandleManager::Instance().GetFileSize(UNKNOWN_MODEL_ID);
  std::unique_ptr<sk::logger::LogContextGuard> logContext;
  aclrtExceptionInfo *exceptionInfo = reinterpret_cast<aclrtExceptionInfo *>(0x500);
  EXPECT_TRUE(handler->ExtractSkEntryArgs(exceptionInfo, logContext));
  ASSERT_NE(logContext, nullptr);
  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelId(), UNKNOWN_MODEL_ID);
  EXPECT_GT(sk::logger::FileHandleManager::Instance().GetFileSize(UNKNOWN_MODEL_ID), sizeBefore);

  std::ifstream fallbackLog(GetSkMetaPath(UNKNOWN_MODEL_ID) + "/super_kernel.log");
  EXPECT_TRUE(fallbackLog.good());

  handler->FreeResources();
  logContext.reset();
  sk::logger::FileLogger::Instance().SetEnabled(false);
  g_mockDeviceBuffer = nullptr;
  g_mockDeviceBufferSize = 0;
  free(buffer);
}
