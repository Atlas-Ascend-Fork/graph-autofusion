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
 * @file aclrt_stubs.cpp
 * @brief Stub implementations for aclrt runtime interfaces used in unit tests
 */

#include "acl/acl.h"
#include "ut_common_stubs.h"
#include "model_fixture_internal.h"
#include "exception_fixture.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

extern "C" {

// Error codes
#ifndef ACL_ERROR_NONE
#define ACL_ERROR_NONE 0
#endif

// Internal RI task structure for stub implementation
using AclmdlRITaskInternal = sk::test::RuntimeTask;

static inline AclmdlRITaskInternal *RITaskToInternal(aclmdlRITask task) {
  return reinterpret_cast<AclmdlRITaskInternal *>(task);
}

// 获取 model id from RI handle. Returns the pointer value cast to uint32, so
// tests can verify a deterministic id without explicit setup.
aclError aclmdlRIGetId(aclmdlRI modelRI, uint32_t *modelId) {
  aclError forcedRet = SkUtGetAclmdlRIGetIdRet();
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  if (modelId == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *modelId = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(modelRI) & 0xFFFFFFFFU);
  return ACL_SUCCESS;
}

// 获取流
aclError aclmdlRIGetStreams(aclmdlRI modelRI, aclrtStream *streams, uint32_t *numStreams) {
  (void)modelRI;
  if (SkUtGetThrowOnAclmdlRIGetStreams() != 0) {
    throw std::runtime_error("ut-injected aclmdlRIGetStreams exception");
  }
  const int phase = (streams == nullptr ? 0 : 1);
  aclError forcedRet = SkUtGetAclmdlRIGetStreamsRet(phase);
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  if (numStreams == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  aclError modelResult;
  if (sk::test::ModelStreams(modelRI, streams, numStreams, modelResult)) {
    return modelResult;
  }
  uint32_t streamNum = SkUtGetModelStreamNum();
  if (streams == nullptr) {
    *numStreams = streamNum;
    return ACL_ERROR_NONE;
  }

  for (uint32_t i = 0; i < streamNum; ++i) {
    streams[i] = reinterpret_cast<aclrtStream>(static_cast<uintptr_t>(i + 1));
  }
  *numStreams = streamNum;
  return ACL_ERROR_NONE;
}

// 获取任务
aclError aclrtStreamGetTasks(aclrtStream stream, aclrtTask *tasks, uint32_t *numTasks) {
  const int phase = (tasks == nullptr ? 0 : 1);
  aclError forcedRet = SkUtGetAclrtStreamGetTasksRet(phase);
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  if (numTasks == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  uint32_t streamIdx = 0;
  if (stream != nullptr) {
    streamIdx = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(stream) - 1U);
  }
  uint32_t taskNum = SkUtGetStreamTaskNum(streamIdx);
  if (tasks == nullptr) {
    *numTasks = taskNum;
    return ACL_ERROR_NONE;
  }

  for (uint32_t i = 0; i < taskNum; ++i) {
    uintptr_t taskHandle = (static_cast<uintptr_t>(streamIdx + 1) << 32U) | static_cast<uintptr_t>(i + 1);
    tasks[i] = reinterpret_cast<aclrtTask>(taskHandle);
  }
  *numTasks = taskNum;
  return ACL_ERROR_NONE;
}

// 获取任务类型
aclError aclrtTaskGetType(aclrtTask task, aclrtTaskType *type) {
  aclError forcedRet = SkUtGetAclrtTaskGetTypeRet();
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  if (type == nullptr || task == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  uintptr_t taskHandle = reinterpret_cast<uintptr_t>(task);
  uint32_t streamIdx = static_cast<uint32_t>((taskHandle >> 32U) & 0xFFFFFFFFU);
  uint32_t taskIdx = static_cast<uint32_t>(taskHandle & 0xFFFFFFFFU);
  if (streamIdx > 0) {
    streamIdx -= 1U;
  }
  if (taskIdx > 0) {
    taskIdx -= 1U;
  }
  *type = SkUtGetTaskType(streamIdx, taskIdx);
  return ACL_ERROR_NONE;
}

// 更新模型资源信息
aclError aclmdlRIUpdate(aclmdlRI modelRI) {
  (void)modelRI;
  aclError forcedRet = SkUtGetAclmdlRIUpdateRet();
  sk::test::RecordUpdate(modelRI, forcedRet);
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  return ACL_ERROR_NONE;
}

// 获取设备
aclError aclrtGetDevice(int32_t *deviceId) {
  aclError forcedRet = SkUtGetAclrtGetDeviceRet();
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  if (deviceId == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *deviceId = 0;
  return ACL_ERROR_NONE;
}

const char *aclrtGetSocName(void) {
  return SkUtGetAclrtGetSocName();
}

aclError aclrtGetDeviceCount(uint32_t *count) {
  *count = SkUtGetDeviceCount();
  return SkUtGetAclrtGetDeviceCountRet();
}

// 获取设备信息
aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value) {
  (void)deviceId;
  aclError forcedRet = SkUtGetAclrtGetDeviceInfoRet();
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  if (value == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (attr == ACL_DEV_ATTR_CUBE_CORE_NUM) {
    *value = 32;
  } else if (attr == ACL_DEV_ATTR_VECTOR_CORE_NUM) {
    *value = 32;
  }
  return ACL_ERROR_NONE;
}

// 从二进制获取函数
aclError aclrtBinaryGetFunction(aclrtBinHandle binHdl, const char *funcName, aclrtFuncHandle *funcHdl) {
  if (funcName == nullptr || funcHdl == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  SkUtSetLastBinaryGetFunctionName(funcName);
  if (SkUtGetBinaryGetFunctionNullHandle() != 0) {
    *funcHdl = nullptr;
    return ACL_ERROR_NONE;
  }
  *funcHdl = sk::test::ResolveFunction(funcName);
  return ACL_ERROR_NONE;
}

// 获取函数地址
aclError aclrtGetFunctionAddr(aclrtFuncHandle funcHdl, void **addrAicore, void **addrAiv) {
  if (addrAicore == nullptr || addrAiv == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  (void)sk::test::FunctionAddress(funcHdl, addrAicore, addrAiv);
  return ACL_ERROR_NONE;
}

// 获取 kernel 参数句柄的内存大小
aclError aclrtKernelArgsGetHandleMemSize(aclrtFuncHandle funcHdl, size_t *memSize) {
  if (memSize == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *memSize = 1024;
  return ACL_ERROR_NONE;
}

// 获取 kernel 参数的内存大小
aclError aclrtKernelArgsGetMemSize(aclrtFuncHandle funcHdl, size_t argsSize, size_t *devArgsSize) {
  if (devArgsSize == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *devArgsSize = 1024;
  return ACL_ERROR_NONE;
}

// 通过用户内存初始化 kernel 参数
aclError aclrtKernelArgsInitByUserMem(aclrtFuncHandle funcHdl, aclrtArgsHandle argsHdl, void *devArgs,
                                      size_t devArgsSize) {
  if (argsHdl == nullptr || devArgs == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  return ACL_ERROR_NONE;
}

// 添加占位符
aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHdl, aclrtParamHandle *phdl) {
  if (argsHdl == nullptr || phdl == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *phdl = reinterpret_cast<aclrtParamHandle>(0x2000);
  return ACL_ERROR_NONE;
}

// 获取占位符缓冲区
aclError aclrtKernelArgsGetPlaceHolderBuffer(aclrtArgsHandle argsHdl, aclrtParamHandle phdl, size_t bufferSize,
                                             void **buffer) {
  if (argsHdl == nullptr || buffer == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  static char placeholderBuffer[4096];
  *buffer = placeholderBuffer;
  return ACL_ERROR_NONE;
}

// 完成 kernel 参数设置
aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHdl) {
  if (argsHdl == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  return ACL_ERROR_NONE;
}

// 获取函数名称
aclError aclrtGetFunctionName(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name) {
  if (name == nullptr || maxLen == 0) {
    return ACL_ERROR_INVALID_PARAM;
  }
  std::string functionName;
  if (sk::test::FunctionName(funcHandle, functionName)) {
    if (functionName.size() >= maxLen) {
      return ACL_ERROR_INVALID_PARAM;
    }
    std::memcpy(name, functionName.c_str(), functionName.size() + 1);
  }
  return ACL_ERROR_NONE;
}

// 内存复制
aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
  if (SkUtFailAclrtMemcpy()) {
    return ACL_ERROR_FAILURE;
  }
  if (dst == nullptr || src == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (count > destMax) {
    return ACL_ERROR_INVALID_PARAM;
  }
  (void)sk::test::CopyRegisteredMemory(dst, src, count, kind);
  return ACL_ERROR_NONE;
}

// 内存设置
aclError aclrtMemset(void *devPtr, size_t maxCount, int value, size_t count) {
  aclError ctrlRet = SkUtGetAclrtMemsetRet();
  if (ctrlRet != ACL_SUCCESS) {
    return ctrlRet;
  }
  if (devPtr == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (count > maxCount) {
    return ACL_ERROR_INVALID_PARAM;
  }
  sk::test::SetRegisteredMemory(devPtr, value, count);
  return ACL_ERROR_NONE;
}

// 获取二进制设备地址
aclError aclrtBinaryGetDevAddress(aclrtBinHandle binHdl, void **devAddr, size_t *devSize) {
  if (devAddr == nullptr || devSize == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *devAddr = nullptr;
  *devSize = 0;
  (void)sk::test::BinaryAddress(binHdl, devAddr, devSize);
  return ACL_ERROR_NONE;
}

aclError aclrtGetFunctionAttribute(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType, int64_t *attrValue) {
  if (attrValue == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (sk::test::FunctionAttribute(funcHandle, attrType, *attrValue)) {
    return ACL_SUCCESS;
  }
  if (attrType == ACL_FUNC_ATTR_KERNEL_TYPE) {
    *attrValue = 0;  // 默认内核类型
  } else {
    *attrValue = 0;
  }
  return ACL_ERROR_NONE;
}

aclError aclrtFunctionGetAvailDynUbufPerBlock(void *func, uint32_t flags, size_t *dynamicUbufSize) {
  (void)func;
  (void)flags;
  aclError forcedRet = SkUtGetAclrtFunctionGetAvailDynUbufPerBlockRet();
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  if (dynamicUbufSize == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *dynamicUbufSize = SkUtGetAclrtFunctionAvailDynUbufSize();
  return ACL_SUCCESS;
}

// Memory management
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
  (void)policy;
  if (devPtr == nullptr || size == 0) {
    return ACL_ERROR_INVALID_PARAM;
  }
  aclError forcedRet = SkUtGetAclrtMallocRet();
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  // G.RES.02-CPP: 内存申请前，必须对申请内存大小进行合法性校验
  if (size == 0 || size > 1024 * 1024 * 1024) {  // Limit to 1GB
    return ACL_ERROR_INVALID_PARAM;
  }
  *devPtr = malloc(size);
  if (*devPtr == nullptr) {
    return ACL_ERROR_FAILURE;
  }
  sk::test::TrackAllocation(*devPtr, size);
  return ACL_ERROR_NONE;
}

aclError aclrtFree(void *devPtr) {
  aclError forcedRet = SkUtGetAclrtFreeRet();
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  if (devPtr != nullptr) {
    sk::test::ForgetAllocation(devPtr);
    free(devPtr);
  }
  return ACL_ERROR_NONE;
}

aclError aclrtMallocHost(void **hostPtr, size_t size) {
  if (SkUtFailAclrtMallocHost()) {
    return ACL_ERROR_FAILURE;
  }
  if (hostPtr == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  // G.RES.02-CPP: 内存申请前，必须对申请内存大小进行合法性校验
  if (size == 0 || size > 1024 * 1024 * 1024) {  // Limit to 1GB
    return ACL_ERROR_INVALID_PARAM;
  }
  *hostPtr = malloc(size);
  if (*hostPtr == nullptr) {
    return ACL_ERROR_FAILURE;
  }
  sk::test::TrackAllocation(*hostPtr, size);
  return ACL_ERROR_NONE;
}

aclError aclrtFreeHost(void *hostPtr) {
  if (hostPtr != nullptr) {
    sk::test::ForgetAllocation(hostPtr);
    free(hostPtr);
  }
  return ACL_ERROR_NONE;
}

aclError aclmdlRIDestroyRegisterCallback(aclmdlRI modelRI, aclmdlRIDestroyCallbackFunc callback, void *userData) {
  aclError forcedRet = SkUtGetAclmdlRIDestroyRegisterCallbackRet();
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  if (modelRI == nullptr || callback == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  return SkUtRegisterModelDestroyCallback(modelRI, callback, userData);
}

// Exception handling
aclError aclrtExceptionInfoCallbackRegister(aclrtExceptionInfoCallbackFunc callback) {
  if (callback == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  sk::test::RegisterExceptionCallback(callback);
  return ACL_ERROR_NONE;
}

aclError aclrtGetFuncHandleFromExceptionInfo(const aclrtExceptionInfo *exceptionInfo, aclrtFuncHandle *funcHandle) {
  if (exceptionInfo == nullptr || funcHandle == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (auto *exception = sk::test::FindException(exceptionInfo)) {
    *funcHandle = exception->function;
    return exception->functionResult;
  }
  // Preserve legacy UT behavior for exceptions not owned by the fixture.
  *funcHandle = reinterpret_cast<aclrtFuncHandle>(0x3000);
  return ACL_ERROR_NONE;
}

aclError aclrtGetArgsFromExceptionInfo(const aclrtExceptionInfo *exceptionInfo, void **args, uint32_t *argsLen) {
  if (exceptionInfo == nullptr || args == nullptr || argsLen == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (auto *exception = sk::test::FindException(exceptionInfo)) {
    *args = exception->args;
    *argsLen = exception->argsSize;
    return exception->argsResult;
  }
  // Preserve legacy UT behavior for exceptions not owned by the fixture.
  *args = nullptr;
  *argsLen = 0;
  return ACL_ERROR_NONE;
}

// ==================== RI Task API Stubs ====================

aclError aclmdlRITaskGetType(aclmdlRITask task, aclmdlRITaskType *type) {
  if (type == nullptr || task == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  AclmdlRITaskInternal *internal = RITaskToInternal(task);
  *type = internal->type;
  return ACL_ERROR_NONE;
}

aclError aclmdlRITaskGetParams(aclmdlRITask task, aclmdlRITaskParams *params) {
  if (params == nullptr || task == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  AclmdlRITaskInternal *internal = RITaskToInternal(task);
  *params = internal->params;
  return ACL_ERROR_NONE;
}

aclError aclmdlRITaskSetParams(aclmdlRITask task, aclmdlRITaskParams *params) {
  if (params == nullptr || task == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (!sk::test::ValidTaskUpdate(task, *params)) {
    return ACL_ERROR_INVALID_PARAM;
  }
  AclmdlRITaskInternal *internal = RITaskToInternal(task);
  internal->params = *params;
  sk::test::RecordTaskUpdate(task, *params);
  return ACL_ERROR_NONE;
}

aclError aclmdlRITaskDisable(aclmdlRITask task) {
  if (task == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  sk::test::RecordDisable(task);
  return ACL_ERROR_NONE;
}

aclError aclmdlRITaskGetSeqId(aclmdlRITask task, uint32_t *id) {
  if (id == nullptr || task == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  AclmdlRITaskInternal *internal = RITaskToInternal(task);
  *id = internal->task_id;
  return ACL_ERROR_NONE;
}

aclError aclmdlRIKernelTaskGetAttribute(aclmdlRITask task, aclrtLaunchKernelAttrId attrId,
                                        aclrtLaunchKernelAttrValue *attrValue) {
  if (task == nullptr || attrValue == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }

  AclmdlRITaskInternal *internal = RITaskToInternal(task);
  if (internal->type != ACL_MODEL_RI_TASK_KERNEL) {
    return ACL_ERROR_INVALID_PARAM;
  }

  std::memset(attrValue, 0, sizeof(*attrValue));
  if (attrId != ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE) {
    return ACL_ERROR_NONE;
  }

  aclrtLaunchKernelCfg *cfg = internal->params.kernelTaskParams.cfg;
  if (cfg == nullptr || cfg->attrs == nullptr || cfg->numAttrs == 0) {
    attrValue->schemMode = 0;
    return ACL_ERROR_NONE;
  }

  for (size_t attrIdx = 0; attrIdx < cfg->numAttrs; ++attrIdx) {
    const aclrtLaunchKernelAttr &launchAttr = cfg->attrs[attrIdx];
    if (launchAttr.id != attrId) {
      continue;
    }
    *attrValue = launchAttr.value;
    return ACL_ERROR_NONE;
  }

  attrValue->schemMode = 0;
  return ACL_ERROR_NONE;
}

aclError aclmdlRIGetTasksByStream(aclrtStream stream, aclmdlRITask *tasks, uint32_t *numTasks) {
  if (numTasks == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  aclError modelResult;
  if (sk::test::StreamTasks(stream, tasks, numTasks, modelResult)) {
    return modelResult;
  }

  uint32_t streamIdx = 0;
  if (stream != nullptr) {
    streamIdx = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(stream) - 1U);
  }
  uint32_t taskNum = SkUtGetStreamTaskNum(streamIdx);
  if (tasks == nullptr) {
    *numTasks = taskNum;
    return ACL_ERROR_NONE;
  }

  static std::vector<std::vector<AclmdlRITaskInternal>> modelRiTasks;
  if (modelRiTasks.size() <= streamIdx) {
    modelRiTasks.resize(streamIdx + 1U);
  }
  auto &streamTasks = modelRiTasks[streamIdx];
  streamTasks.resize(taskNum);
  for (uint32_t taskIdx = 0; taskIdx < taskNum; ++taskIdx) {
    auto &task = streamTasks[taskIdx];
    task.task_id = (streamIdx << 16U) + taskIdx;
    task.type = ACL_MODEL_RI_TASK_DEFAULT;
    task.params = {};
    tasks[taskIdx] = reinterpret_cast<aclmdlRITask>(&task);
  }
  *numTasks = taskNum;
  return ACL_ERROR_NONE;
}

aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode) {
  if (mode == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (*mode == ACL_MODEL_RI_CAPTURE_MODE_GLOBAL) {
    *mode = ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
  } else {
    *mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
  }
  return ACL_ERROR_NONE;
}

aclError aclrtFunctionGetBinary(aclrtFuncHandle funcHandle, aclrtBinHandle *binHandle) {
  if (binHandle == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *binHandle = nullptr;
  (void)sk::test::FunctionBinary(funcHandle, *binHandle);
  return ACL_ERROR_NONE;
}

// Stub implementations for SkEventRecorder tests
static char g_stubDeviceMemory[1024 * 1024];  // 1MB stub memory

aclError aclrtSetDevice(int32_t deviceId) {
  (void)deviceId;
  return ACL_SUCCESS;
}

// 获取流ID
aclError aclrtStreamGetId(aclrtStream stream, int32_t *streamId) {
  aclError forcedRet = SkUtGetAclrtStreamGetIdRet();
  if (forcedRet != ACL_SUCCESS) {
    return forcedRet;
  }
  if (streamId == nullptr || stream == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (sk::test::StreamId(stream, *streamId)) {
    return ACL_SUCCESS;
  }
  uint32_t streamIdx = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(stream) - 1U);
  *streamId = SkUtGetStreamId(streamIdx);
  return ACL_ERROR_NONE;
}

}  // extern "C"
