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
 * @file ut_common_stubs.cpp
 * @brief Common stub implementations used by aot unit tests
 */

#include "acl/acl.h"
#include "runtime/kernel.h"
#include "sk_common.h"
#include "model_fixture_internal.h"
#include "dlog_pub.h"
#include <chrono>
#include <deque>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <atomic>
#include "ut_common_stubs.h"

ut_log::LogBuffer &ut_log::LogBuffer::Instance() {
  static LogBuffer instance;
  return instance;
}

namespace {

int g_entryBinHandleNull = 0;
int g_securecMemcpyFailOnCall = -1;
int g_securecMemcpySeen = 0;
int g_securecMemsetFailOnCall = -1;
int g_securecMemsetSeen = 0;
std::atomic<int> g_aclrtMemcpyFailOnCall{0};
std::atomic<int> g_aclrtMemcpySeen{0};
std::atomic<int> g_aclrtMallocHostFailOnCall{0};
std::atomic<int> g_aclrtMallocHostSeen{0};

aclError g_aclmdlRIGetStreamsRet[2] = {ACL_SUCCESS, ACL_SUCCESS};
aclError g_aclrtStreamGetTasksRet[2] = {ACL_SUCCESS, ACL_SUCCESS};
aclError g_aclrtTaskGetTypeRet = ACL_SUCCESS;
aclError g_aclrtGetDeviceRet = ACL_SUCCESS;
uint32_t g_deviceCount = 0;
aclError g_aclrtGetDeviceCountRet = ACL_SUCCESS;
aclError g_aclrtGetDeviceInfoRet = ACL_SUCCESS;
aclError g_aclmdlRIUpdateRet = ACL_SUCCESS;
aclError g_aclmdlRIGetIdRet = ACL_SUCCESS;
aclError g_aclmdlRIDestroyRegisterCallbackRet = ACL_SUCCESS;
aclError g_aclrtMallocRet = ACL_SUCCESS;
aclError g_aclrtFreeRet = ACL_SUCCESS;
aclError g_aclrtMemsetRet = ACL_SUCCESS;
aclError g_aclrtStreamGetIdRet = ACL_SUCCESS;
aclError g_aclrtFunctionGetAvailDynUbufPerBlockRet = ACL_SUCCESS;
size_t g_aclrtFunctionAvailDynUbufSize = 0;
int g_throwOnAclmdlRIGetStreams = 0;
int g_binaryGetFunctionNullHandle = 0;
uint32_t g_destroyRegisterCallbackDelayUs = 0;
uint32_t g_destroyRegisterCallbackCallCount = 0;
const char *g_aclrtGetSocName = "Ascend910B";
std::string g_lastBinaryGetFunctionName;

std::unordered_map<aclmdlRI, std::deque<std::pair<aclmdlRIDestroyCallbackFunc, void *>>> g_modelDestroyCallbacks;
uint32_t g_streamNum = 0;
std::vector<uint32_t> g_streamTaskNums;
std::vector<std::vector<aclrtTaskType>> g_taskTypes;
std::vector<int32_t> g_streamIds;
std::vector<std::string> g_debugJsonPrintPaths;

void EnsureStreamStorage(uint32_t streamIdx) {
  if (g_streamTaskNums.size() <= streamIdx) {
    g_streamTaskNums.resize(streamIdx + 1, 0);
  }
  if (g_taskTypes.size() <= streamIdx) {
    g_taskTypes.resize(streamIdx + 1);
  }
}

void EnsureTaskStorage(uint32_t streamIdx, uint32_t taskIdx) {
  EnsureStreamStorage(streamIdx);
  if (g_taskTypes[streamIdx].size() <= taskIdx) {
    g_taskTypes[streamIdx].resize(taskIdx + 1, ACL_RT_TASK_KERNEL);
  }
}

}  // namespace

extern "C" {

void SkUtResetCommonStubControls() {
  g_entryBinHandleNull = 0;
  g_securecMemcpyFailOnCall = -1;
  g_securecMemcpySeen = 0;
  g_securecMemsetFailOnCall = -1;
  g_securecMemsetSeen = 0;

  g_aclmdlRIGetStreamsRet[0] = ACL_SUCCESS;
  g_aclmdlRIGetStreamsRet[1] = ACL_SUCCESS;
  g_aclrtStreamGetTasksRet[0] = ACL_SUCCESS;
  g_aclrtStreamGetTasksRet[1] = ACL_SUCCESS;
  g_aclrtTaskGetTypeRet = ACL_SUCCESS;
  g_aclrtGetDeviceRet = ACL_SUCCESS;
  g_deviceCount = 0;
  g_aclrtGetDeviceCountRet = ACL_SUCCESS;
  g_aclrtGetDeviceInfoRet = ACL_SUCCESS;
  g_aclmdlRIUpdateRet = ACL_SUCCESS;
  g_aclmdlRIGetIdRet = ACL_SUCCESS;
  g_aclmdlRIDestroyRegisterCallbackRet = ACL_SUCCESS;
  g_aclrtMallocRet = ACL_SUCCESS;
  g_aclrtFreeRet = ACL_SUCCESS;
  g_aclrtMemsetRet = ACL_SUCCESS;
  g_aclrtStreamGetIdRet = ACL_SUCCESS;
  g_aclrtFunctionGetAvailDynUbufPerBlockRet = ACL_SUCCESS;
  g_aclrtFunctionAvailDynUbufSize = 0;
  g_throwOnAclmdlRIGetStreams = 0;
  g_binaryGetFunctionNullHandle = 0;
  g_destroyRegisterCallbackDelayUs = 0;
  g_destroyRegisterCallbackCallCount = 0;
  g_aclrtGetSocName = "Ascend910B";
  g_lastBinaryGetFunctionName.clear();
  g_modelDestroyCallbacks.clear();

  g_streamNum = 0;
  g_streamTaskNums.clear();
  g_taskTypes.clear();
  g_streamIds.clear();
  g_debugJsonPrintPaths.clear();
  SetFunctionAllocUbufSize(0);
  SetSimtAivType(0);
  SetRtFunctionGetMetaInfoRet(0);
}

void SkUtResetTestControls() {
  SkUtSetAclrtMemcpyFailOnCall(0);
  SkUtSetAclrtMallocHostFailOnCall(0);
  SkUtResetCommonStubControls();
}

void SkUtSetAclrtMemcpyFailOnCall(int call) {
  g_aclrtMemcpySeen = 0;
  g_aclrtMemcpyFailOnCall = call;
}
void SkUtSetAclrtMallocHostFailOnCall(int call) {
  g_aclrtMallocHostSeen = 0;
  g_aclrtMallocHostFailOnCall = call;
}
bool SkUtFailAclrtMemcpy() {
  return g_aclrtMemcpySeen.fetch_add(1) + 1 == g_aclrtMemcpyFailOnCall.load();
}
bool SkUtFailAclrtMallocHost() {
  return g_aclrtMallocHostSeen.fetch_add(1) + 1 == g_aclrtMallocHostFailOnCall.load();
}

void SkUtSetAclmdlRIGetStreamsRet(int phase, aclError ret) {
  if (phase < 0 || phase > 1) {
    return;
  }
  g_aclmdlRIGetStreamsRet[phase] = ret;
}

void SkUtSetAclrtStreamGetTasksRet(int phase, aclError ret) {
  if (phase < 0 || phase > 1) {
    return;
  }
  g_aclrtStreamGetTasksRet[phase] = ret;
}

void SkUtSetAclrtTaskGetTypeRet(aclError ret) {
  g_aclrtTaskGetTypeRet = ret;
}

void SkUtSetAclrtGetDeviceRet(aclError ret) {
  g_aclrtGetDeviceRet = ret;
}

void SkUtSetDeviceCount(uint32_t count) {
  g_deviceCount = count;
}

uint32_t SkUtGetDeviceCount() {
  return g_deviceCount;
}

void SkUtSetAclrtGetDeviceCountRet(aclError ret) {
  g_aclrtGetDeviceCountRet = ret;
}

aclError SkUtGetAclrtGetDeviceCountRet() {
  return g_aclrtGetDeviceCountRet;
}

void SkUtSetAclrtGetDeviceInfoRet(aclError ret) {
  g_aclrtGetDeviceInfoRet = ret;
}

void SkUtSetAclmdlRIUpdateRet(aclError ret) {
  g_aclmdlRIUpdateRet = ret;
}

void SkUtSetAclmdlRIGetIdRet(aclError ret) {
  g_aclmdlRIGetIdRet = ret;
}

void SkUtSetAclmdlRIDestroyRegisterCallbackRet(aclError ret) {
  g_aclmdlRIDestroyRegisterCallbackRet = ret;
}

void SkUtSetAclrtMallocRet(aclError ret) {
  g_aclrtMallocRet = ret;
}

void SkUtSetAclrtFreeRet(aclError ret) {
  g_aclrtFreeRet = ret;
}

void SkUtSetAclrtStreamGetIdRet(aclError ret) {
  g_aclrtStreamGetIdRet = ret;
}

void SkUtSetAclrtFunctionGetAvailDynUbufPerBlockRet(aclError ret) {
  g_aclrtFunctionGetAvailDynUbufPerBlockRet = ret;
}

void SkUtSetAclrtFunctionAvailDynUbufSize(size_t dynUbufSize) {
  g_aclrtFunctionAvailDynUbufSize = dynUbufSize;
}

void SkUtSetThrowOnAclmdlRIGetStreams(int enable) {
  g_throwOnAclmdlRIGetStreams = enable;
}

void SkUtSetDestroyRegisterCallbackDelayUs(uint32_t delayUs) {
  g_destroyRegisterCallbackDelayUs = delayUs;
}

void SkUtSetBinaryGetFunctionNullHandle(int enable) {
  g_binaryGetFunctionNullHandle = enable;
}

aclError SkUtGetAclmdlRIGetStreamsRet(int phase) {
  if (phase < 0 || phase > 1) {
    return ACL_SUCCESS;
  }
  return g_aclmdlRIGetStreamsRet[phase];
}

aclError SkUtGetAclrtStreamGetTasksRet(int phase) {
  if (phase < 0 || phase > 1) {
    return ACL_SUCCESS;
  }
  return g_aclrtStreamGetTasksRet[phase];
}

aclError SkUtGetAclrtTaskGetTypeRet() {
  return g_aclrtTaskGetTypeRet;
}

aclError SkUtGetAclrtGetDeviceRet() {
  return g_aclrtGetDeviceRet;
}

aclError SkUtGetAclrtGetDeviceInfoRet() {
  return g_aclrtGetDeviceInfoRet;
}

aclError SkUtGetAclmdlRIUpdateRet() {
  return g_aclmdlRIUpdateRet;
}

aclError SkUtGetAclmdlRIGetIdRet() {
  return g_aclmdlRIGetIdRet;
}

aclError SkUtGetAclmdlRIDestroyRegisterCallbackRet() {
  return g_aclmdlRIDestroyRegisterCallbackRet;
}

aclError SkUtGetAclrtMallocRet() {
  return g_aclrtMallocRet;
}

aclError SkUtGetAclrtFreeRet() {
  return g_aclrtFreeRet;
}

void SkUtSetAclrtMemsetRet(aclError ret) {
  g_aclrtMemsetRet = ret;
}

aclError SkUtGetAclrtMemsetRet() {
  return g_aclrtMemsetRet;
}

aclError SkUtGetAclrtStreamGetIdRet() {
  return g_aclrtStreamGetIdRet;
}

aclError SkUtGetAclrtFunctionGetAvailDynUbufPerBlockRet() {
  return g_aclrtFunctionGetAvailDynUbufPerBlockRet;
}

size_t SkUtGetAclrtFunctionAvailDynUbufSize() {
  return g_aclrtFunctionAvailDynUbufSize;
}

int SkUtGetThrowOnAclmdlRIGetStreams() {
  return g_throwOnAclmdlRIGetStreams;
}

uint32_t SkUtGetDestroyRegisterCallbackCallCount() {
  return g_destroyRegisterCallbackCallCount;
}

int SkUtGetBinaryGetFunctionNullHandle() {
  return g_binaryGetFunctionNullHandle;
}

const char *SkUtGetLastBinaryGetFunctionName() {
  return g_lastBinaryGetFunctionName.c_str();
}

aclError SkUtRegisterModelDestroyCallback(aclmdlRI modelRI, aclmdlRIDestroyCallbackFunc callback, void *userData) {
  if (modelRI == nullptr || callback == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  ++g_destroyRegisterCallbackCallCount;
  if (g_destroyRegisterCallbackDelayUs > 0U) {
    std::this_thread::sleep_for(std::chrono::microseconds(g_destroyRegisterCallbackDelayUs));
  }
  g_modelDestroyCallbacks[modelRI].emplace_back(callback, userData);
  return ACL_SUCCESS;
}

aclError SkUtInvokeModelDestroyCallback(aclmdlRI modelRI) {
  auto it = g_modelDestroyCallbacks.find(modelRI);
  if (it == g_modelDestroyCallbacks.end() || it->second.empty()) {
    return ACL_ERROR_INVALID_PARAM;
  }
  auto callbackInfo = it->second.front();
  it->second.pop_front();
  if (it->second.empty()) {
    g_modelDestroyCallbacks.erase(it);
  }
  auto callback = callbackInfo.first;
  void *userData = callbackInfo.second;
  callback(userData);
  return ACL_SUCCESS;
}

size_t SkUtGetModelDestroyCallbackCount() {
  size_t callbackCount = 0U;
  for (const auto &callbackEntry : g_modelDestroyCallbacks) {
    callbackCount += callbackEntry.second.size();
  }
  return callbackCount;
}

void SkUtSetModelStreamNum(uint32_t streamNum) {
  g_streamNum = streamNum;
  if (g_streamTaskNums.size() < streamNum) {
    g_streamTaskNums.resize(streamNum, 0);
  }
  if (g_taskTypes.size() < streamNum) {
    g_taskTypes.resize(streamNum);
  }
  if (g_streamIds.size() < streamNum) {
    g_streamIds.resize(streamNum, -1);
  }
}

uint32_t SkUtGetModelStreamNum() {
  return g_streamNum;
}

void SkUtSetStreamTaskNum(uint32_t streamIdx, uint32_t taskNum) {
  EnsureStreamStorage(streamIdx);
  g_streamTaskNums[streamIdx] = taskNum;
  g_taskTypes[streamIdx].resize(taskNum, ACL_RT_TASK_KERNEL);
}

uint32_t SkUtGetStreamTaskNum(uint32_t streamIdx) {
  if (streamIdx >= g_streamTaskNums.size()) {
    return 0;
  }
  return g_streamTaskNums[streamIdx];
}

void SkUtSetTaskType(uint32_t streamIdx, uint32_t taskIdx, aclrtTaskType type) {
  EnsureTaskStorage(streamIdx, taskIdx);
  g_taskTypes[streamIdx][taskIdx] = type;
}

aclrtTaskType SkUtGetTaskType(uint32_t streamIdx, uint32_t taskIdx) {
  if (streamIdx >= g_taskTypes.size() || taskIdx >= g_taskTypes[streamIdx].size()) {
    return ACL_RT_TASK_KERNEL;
  }
  return g_taskTypes[streamIdx][taskIdx];
}

void SkUtSetEntryBinHandleNull(int enable) {
  g_entryBinHandleNull = enable;
}

void SkUtSetLastBinaryGetFunctionName(const char *funcName) {
  g_lastBinaryGetFunctionName = funcName == nullptr ? "" : funcName;
}

void SkUtSetSecurecMemcpyFailOnCall(int hitOnCall) {
  g_securecMemcpyFailOnCall = hitOnCall;
  g_securecMemcpySeen = 0;
}

void SkUtSetSecurecMemsetFailOnCall(int hitOnCall) {
  g_securecMemsetFailOnCall = hitOnCall;
  g_securecMemsetSeen = 0;
}

int SkUtSecurecShouldFailMemcpy() {
  if (g_securecMemcpyFailOnCall <= 0) {
    return 0;
  }
  ++g_securecMemcpySeen;
  return g_securecMemcpySeen == g_securecMemcpyFailOnCall ? 1 : 0;
}

int SkUtSecurecShouldFailMemset() {
  if (g_securecMemsetFailOnCall <= 0) {
    return 0;
  }
  ++g_securecMemsetSeen;
  return g_securecMemsetSeen == g_securecMemsetFailOnCall ? 1 : 0;
}

void SkUtSetStreamId(uint32_t streamIdx, int32_t streamId) {
  if (g_streamIds.size() <= streamIdx) {
    g_streamIds.resize(streamIdx + 1, -1);
  }
  g_streamIds[streamIdx] = streamId;
}

int32_t SkUtGetStreamId(uint32_t streamIdx) {
  if (streamIdx >= g_streamIds.size()) {
    return -1;
  }
  return g_streamIds[streamIdx];
}

extern "C" aclrtBinHandle AscendGetEntryBinHandle() {
  if (g_entryBinHandleNull != 0) {
    return nullptr;
  }
  return reinterpret_cast<aclrtBinHandle>(0x1234);
}

void sk_scope_kernel_begin_do_dav_2201(void *stream, ScopeKernelArgs args) {
  sk::test::RecordLaunch(stream, "sk_scope_kernel_begin_dav_2201", args.name, sizeof(args));
}

void sk_scope_kernel_end_do_dav_2201(void *stream, ScopeKernelArgs args) {
  sk::test::RecordLaunch(stream, "sk_scope_kernel_end_dav_2201", args.name, sizeof(args));
}

void sk_placeholder_kernel_do_dav_2201(void *stream, ScopeKernelArgs args) {
  sk::test::RecordLaunch(stream, "sk_placeholder_kernel_dav_2201", args.name, sizeof(args));
}

void sk_scope_kernel_begin_do_dav_3510(void *stream, ScopeKernelArgs args) {
  sk::test::RecordLaunch(stream, "sk_scope_kernel_begin_dav_3510", args.name, sizeof(args));
}

void sk_scope_kernel_end_do_dav_3510(void *stream, ScopeKernelArgs args) {
  sk::test::RecordLaunch(stream, "sk_scope_kernel_end_dav_3510", args.name, sizeof(args));
}

void sk_placeholder_kernel_do_dav_3510(void *stream, ScopeKernelArgs args) {
  sk::test::RecordLaunch(stream, "sk_placeholder_kernel_dav_3510", args.name, sizeof(args));
}

void SkUtSetAclrtGetSocName(const char *socName) {
  g_aclrtGetSocName = socName;
}

const char *SkUtGetAclrtGetSocName() {
  return g_aclrtGetSocName;
}

uint32_t SkUtGetDebugJsonPrintCallCount() {
  return static_cast<uint32_t>(g_debugJsonPrintPaths.size());
}

const char *SkUtGetDebugJsonPrintPath(uint32_t index) {
  if (index >= g_debugJsonPrintPaths.size()) {
    return nullptr;
  }
  return g_debugJsonPrintPaths[index].c_str();
}

void SkUtRecordDebugJsonPrintPath(const char *path) {
  g_debugJsonPrintPaths.emplace_back(path == nullptr ? "" : path);
}

}  // extern "C"
