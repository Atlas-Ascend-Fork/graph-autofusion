/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_ACL_ACL_RT_H_
#define INC_EXTERNAL_ACL_ACL_RT_H_

#include <stdint.h>
#include <stddef.h>
#include "runtime/base.h"

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
static const int ACL_SUCCESS = 0;
static const int ACL_ERROR_NONE = 0;
static const int ACL_ERROR_FAILURE = 500001;
static const int ACL_ERROR_INVALID_PARAM = 100000;

// EOK for secure functions
#ifndef EOK
#define EOK 0
#endif

#define ACL_EVENT_EXTERNAL 0x00000020U
#define ACL_EVENT_RECORD_DEFAULT 0x0U
#define ACL_EVENT_RECORD_EXTERNAL 0x01U
#define ACL_EVENT_WAIT_DEFAULT 0x0U
#define ACL_EVENT_WAIT_EXTERNAL 0x01U

// ACL data types
#define ACL_UINT8 2
#define ACL_INT8 3
#define ACL_INT32 4
#define ACL_FP16 5
#define ACL_FP32 6
#define ACL_FP64 7

// ACL format types
#define ACL_FORMAT_ND 0
#define ACL_FORMAT_NCHW 1
#define ACL_FORMAT_NHWC 2
#define ACL_FORMAT_CHWN 3

// Basic types
typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *aclrtContext;
typedef void *aclrtNotify;
typedef void *aclrtTaskGrp;
typedef void *aclrtFuncHandle;
typedef void *aclrtBinHandle;

typedef void *aclrtArgsHandle;
typedef void *aclrtParamHandle;
typedef void *aclmdlRI;
typedef void *aclrtTask;  // void* as per user request
typedef int aclError;

typedef enum aclrtLaunchKernelAttrId {
  ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE = 1,
  ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE = 2,
  ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE = 3,
  ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET = 4,
  ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH = 5,
  ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP = 6,
  ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT = 7,
  // ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT and ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US cannot be carried at the same time
  ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US = 8,
} aclrtLaunchKernelAttrId;

typedef union aclrtLaunchKernelAttrValue {
  uint8_t schemMode;
  uint32_t localMemorySize;
  uint32_t dynUBufSize;
  uint8_t engineType;
  uint8_t isBlockTaskPrefetch;
  uint8_t isDataDump;
  uint16_t timeout;
  uint32_t rsv[4];
} aclrtLaunchKernelAttrValue;

typedef struct aclrtLaunchKernelAttr {
  aclrtLaunchKernelAttrId id;
  aclrtLaunchKernelAttrValue value;
} aclrtLaunchKernelAttr;

typedef struct aclrtLaunchKernelCfg {
  aclrtLaunchKernelAttr *attrs;
  size_t numAttrs;
} aclrtLaunchKernelCfg;

// Exception info structure - defines expandInfo with type field used in exception handling
typedef struct aclrtExceptionInfo {
  void *reserved[16];  // Reserved fields
  struct {
    rtExceptionExpandType_t type;  // exception type (AICORE, FFTS_PLUS, FUSION, etc.)
  } expandInfo;
} aclrtExceptionInfo;

// Exception callback function type
typedef void (*aclrtExceptionInfoCallbackFunc)(aclrtExceptionInfo *exceptionInfo);
typedef void (*aclmdlRIDestroyCallbackFunc)(void *userData);

typedef enum aclrtMemMallocPolicy {
  ACL_MEM_MALLOC_HUGE_FIRST = 0,
} aclrtMemMallocPolicy;

// Memory copy kind
typedef enum aclrtMemcpyKind {
  ACL_MEMCPY_HOST_TO_HOST,
  ACL_MEMCPY_HOST_TO_DEVICE,
  ACL_MEMCPY_DEVICE_TO_HOST,
  ACL_MEMCPY_DEVICE_TO_DEVICE,
} aclrtMemcpyKind;

// Device attributes
typedef enum aclrtDevAttr {
  ACL_DEV_ATTR_CUBE_CORE_NUM,
  ACL_DEV_ATTR_VECTOR_CORE_NUM,
} aclrtDevAttr;

// Task types
typedef enum aclrtTaskType {
  ACL_RT_TASK_DEFAULT,
  ACL_RT_TASK_KERNEL,
  ACL_RT_TASK_EVENT_RECORD,
  ACL_RT_TASK_EVENT_WAIT,
  ACL_RT_TASK_EVENT_RESET,
  ACL_RT_TASK_VALUE_WRITE,
  ACL_RT_TASK_VALUE_WAIT,
} aclrtTaskType;

typedef enum aclrtTaskFlag {
  ACL_RT_TASK_INVALID,
  ACL_RT_TASK_VALID,
} aclrtTaskFlag;

typedef enum aclrtEventType {
  ACL_RT_EVENT_NORMAL,
  ACL_RT_EVENT_MEMORY,
} aclrtEventType;

// RI Task types (new API)
typedef enum aclmdlRITaskType {
  ACL_MODEL_RI_TASK_DEFAULT,
  ACL_MODEL_RI_TASK_KERNEL,
  ACL_MODEL_RI_TASK_EVENT_RECORD,
  ACL_MODEL_RI_TASK_EVENT_WAIT,
  ACL_MODEL_RI_TASK_EVENT_RESET,
  ACL_MODEL_RI_TASK_VALUE_WRITE,
  ACL_MODEL_RI_TASK_VALUE_WAIT,
} aclmdlRITaskType;

typedef enum aclmdlRICaptureMode {
  ACL_MODEL_RI_CAPTURE_MODE_GLOBAL = 0,
  ACL_MODEL_RI_CAPTURE_MODE_RELAXED,
} aclmdlRICaptureMode;

typedef void *aclmdlRITask;

// RI Task parameter structures (new API)
typedef struct aclmdlRIKernelTaskParams {
  aclrtFuncHandle funcHandle;
  aclrtLaunchKernelCfg *cfg;
  void *args;
  uint32_t isHostArgs;
  size_t argsSize;
  uint32_t numBlocks;
  uint32_t rsv[10];
} aclmdlRIKernelTaskParams;

typedef struct aclmdlRIValueWriteTaskParams {
  void *devAddr;
  uint64_t value;
} aclmdlRIValueWriteTaskParams;

typedef struct aclmdlRIValueWaitTaskParams {
  void *devAddr;
  uint64_t value;
  uint32_t flag;
} aclmdlRIValueWaitTaskParams;

typedef struct aclmdlRIEventRecordTaskParams {
  aclrtEvent event;
  uint32_t eventFlag;
  uint32_t recordFlag;
} aclmdlRIEventRecordTaskParams;

typedef struct aclmdlRIEventWaitTaskParams {
  aclrtEvent event;
  uint32_t eventFlag;
  uint32_t waitFlag;
} aclmdlRIEventWaitTaskParams;

typedef struct aclmdlRIEventResetTaskParams {
  aclrtEvent event;
  uint32_t eventFlag;
  uint32_t resetFlag;
} aclmdlRIEventResetTaskParams;

typedef struct aclmdlRITaskParams {
  aclmdlRITaskType type;
  uint32_t reserved0[3];
  aclrtTaskGrp taskGrp;
  void *opInfoPtr;
  size_t opInfoSize;
  char reserved1[32];
  union {
    char reserved2[128];
    struct aclmdlRIKernelTaskParams kernelTaskParams;
    struct aclmdlRIEventRecordTaskParams eventRecordTaskParams;
    struct aclmdlRIEventWaitTaskParams eventWaitTaskParams;
    struct aclmdlRIEventResetTaskParams eventResetTaskParams;
    struct aclmdlRIValueWriteTaskParams valueWriteTaskParams;
    struct aclmdlRIValueWaitTaskParams valueWaitTaskParams;
  };
} aclmdlRITaskParams;

typedef enum {
  ACL_KERNEL_TYPE_AICORE = 0,  // MIX KERNEL
  ACL_KERNEL_TYPE_CUBE = 1,    // AI CUBE CORE
  ACL_KERNEL_TYPE_VECTOR = 2,  // AI VECTOR CORE
  ACL_KERNEL_TYPE_MIX = 3,
  ACL_KERNEL_TYPE_AICPU = 100,
} aclrtKernelType;

typedef enum {
  ACL_FUNC_ATTR_KERNEL_TYPE = 1,
  ACL_FUNC_ATTR_KERNEL_RATIO = 2,
  ACL_FUNC_ATTR_KERNEL_SCHED_MODE = 3,
} aclrtFuncAttribute;

// Function declarations
aclError aclmdlRIGetStreams(aclmdlRI modelRI, aclrtStream *streams, uint32_t *numStreams);
aclError aclmdlRIGetId(aclmdlRI modelRI, uint32_t *modelId);
aclError aclrtStreamGetTasks(aclrtStream stream, aclrtTask *tasks, uint32_t *numTasks);
aclError aclrtTaskGetType(aclrtTask task, aclrtTaskType *type);
aclError aclmdlRIUpdate(aclmdlRI modelRI);
aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value);
aclError aclrtGetDevice(int32_t *deviceId);
aclError aclrtGetDeviceCount(uint32_t *count);
const char *aclrtGetSocName(void);
aclError aclrtStreamGetId(aclrtStream stream, int32_t *streamId);
aclError aclmdlRIDebugJsonPrint(aclmdlRI model, const char *path, uint32_t flag);

// RI Task API declarations
aclError aclmdlRITaskGetType(aclmdlRITask task, aclmdlRITaskType *type);
aclError aclmdlRITaskGetParams(aclmdlRITask task, aclmdlRITaskParams *params);
aclError aclmdlRITaskSetParams(aclmdlRITask task, aclmdlRITaskParams *params);
aclError aclmdlRITaskDisable(aclmdlRITask task);
aclError aclmdlRITaskGetSeqId(aclmdlRITask task, uint32_t *id);
aclError aclmdlRIGetTasksByStream(aclrtStream stream, aclmdlRITask *tasks, uint32_t *numTasks);
aclError aclmdlRIKernelTaskGetAttribute(aclmdlRITask task, aclrtLaunchKernelAttrId attrId,
                                        aclrtLaunchKernelAttrValue *attrValue);
aclError aclrtFunctionGetBinary(aclrtFuncHandle funcHandle, aclrtBinHandle *binHandle);
aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode);

// Binary API
aclError aclrtBinaryGetFunction(aclrtBinHandle binHdl, const char *funcName, aclrtFuncHandle *funcHdl);
aclError aclrtGetFunctionAddr(aclrtFuncHandle funcHdl, void **addrAicore, void **addrAiv);
aclError aclrtKernelArgsGetHandleMemSize(aclrtFuncHandle funcHdl, size_t *memSize);
aclError aclrtKernelArgsGetMemSize(aclrtFuncHandle funcHdl, size_t argsSize, size_t *devArgsSize);
aclError aclrtKernelArgsInitByUserMem(aclrtFuncHandle funcHdl, aclrtArgsHandle argsHdl, void *devArgs,
                                      size_t devArgsSize);
aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHdl, aclrtParamHandle *phdl);
aclError aclrtKernelArgsGetPlaceHolderBuffer(aclrtArgsHandle argsHdl, aclrtParamHandle phdl, size_t bufferSize,
                                             void **buffer);
aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHdl);
aclError aclrtGetFunctionName(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name);
aclError aclrtBinaryGetDevAddress(aclrtBinHandle binHdl, void **devAddr, size_t *devSize);
aclError aclrtGetFunctionAttribute(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType, int64_t *attrValue);
aclError aclrtFunctionGetAvailDynUbufPerBlock(void *func, uint32_t flags, size_t *dynamicUbufSize);

// Memory management functions
aclError aclrtSetDevice(int32_t deviceId);
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
aclError aclrtMemset(void *devPtr, size_t destMax, int value, size_t count);
aclError aclrtFree(void *devPtr);

// Memory management
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
aclError aclrtFree(void *devPtr);
aclError aclrtMallocHost(void **hostPtr, size_t size);
aclError aclrtFreeHost(void *hostPtr);
aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
aclError aclrtMemset(void *devPtr, size_t maxCount, int value, size_t count);

aclError aclmdlRIDestroyRegisterCallback(aclmdlRI modelRI, aclmdlRIDestroyCallbackFunc callback, void *userData);

// Exception handling
aclError aclrtExceptionInfoCallbackRegister(aclrtExceptionInfoCallbackFunc callback);
aclError aclrtGetFuncHandleFromExceptionInfo(const aclrtExceptionInfo *exceptionInfo, aclrtFuncHandle *funcHandle);
aclError aclrtGetArgsFromExceptionInfo(const aclrtExceptionInfo *exceptionInfo, void **args, uint32_t *argsLen);

#ifdef __cplusplus
}
#endif

#endif  // INC_EXTERNAL_ACL_ACL_RT_H_
