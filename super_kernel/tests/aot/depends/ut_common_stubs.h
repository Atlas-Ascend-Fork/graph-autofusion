/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SUPER_KERNEL_UT_COMMON_STUBS_H_
#define SUPER_KERNEL_UT_COMMON_STUBS_H_

#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

// UT control hooks used by unit tests and stubs.
void SkUtResetTestControls();
void SkUtSetAclrtMemcpyFailOnCall(int call);
void SkUtSetAclrtMallocHostFailOnCall(int call);
bool SkUtFailAclrtMemcpy();
bool SkUtFailAclrtMallocHost();

void SkUtSetAclmdlRIGetStreamsRet(int phase, aclError ret);
void SkUtSetAclrtStreamGetTasksRet(int phase, aclError ret);
void SkUtSetAclrtTaskGetTypeRet(aclError ret);
void SkUtSetAclrtGetDeviceRet(aclError ret);
void SkUtSetDeviceCount(uint32_t count);
uint32_t SkUtGetDeviceCount();
void SkUtSetAclrtGetDeviceCountRet(aclError ret);
aclError SkUtGetAclrtGetDeviceCountRet();
void SkUtSetAclrtGetDeviceInfoRet(aclError ret);
void SkUtSetAclmdlRIUpdateRet(aclError ret);
void SkUtSetAclmdlRIGetIdRet(aclError ret);
void SkUtSetAclmdlRIDestroyRegisterCallbackRet(aclError ret);
void SkUtSetAclrtMallocRet(aclError ret);
void SkUtSetAclrtFreeRet(aclError ret);
void SkUtSetAclrtMemsetRet(aclError ret);
void SkUtSetAclrtStreamGetIdRet(aclError ret);
void SkUtSetAclrtFunctionGetAvailDynUbufPerBlockRet(aclError ret);
void SkUtSetAclrtFunctionAvailDynUbufSize(size_t dynUbufSize);
void SkUtSetThrowOnAclmdlRIGetStreams(int enable);
void SkUtSetDestroyRegisterCallbackDelayUs(uint32_t delayUs);
int SkUtGetBinaryGetFunctionNullHandle();
const char *SkUtGetLastBinaryGetFunctionName();

aclError SkUtGetAclmdlRIGetStreamsRet(int phase);
aclError SkUtGetAclrtStreamGetTasksRet(int phase);
aclError SkUtGetAclrtTaskGetTypeRet();
aclError SkUtGetAclrtGetDeviceRet();
aclError SkUtGetAclrtGetDeviceInfoRet();
aclError SkUtGetAclmdlRIUpdateRet();
aclError SkUtGetAclmdlRIGetIdRet();
aclError SkUtGetAclmdlRIDestroyRegisterCallbackRet();
aclError SkUtGetAclrtMallocRet();
aclError SkUtGetAclrtFreeRet();
aclError SkUtGetAclrtMemsetRet();
aclError SkUtGetAclrtStreamGetIdRet();
aclError SkUtGetAclrtFunctionGetAvailDynUbufPerBlockRet();
size_t SkUtGetAclrtFunctionAvailDynUbufSize();
int SkUtGetThrowOnAclmdlRIGetStreams();
uint32_t SkUtGetDestroyRegisterCallbackCallCount();
aclError SkUtRegisterModelDestroyCallback(aclmdlRI modelRI, aclmdlRIDestroyCallbackFunc callback, void *userData);

aclError SkUtInvokeModelDestroyCallback(aclmdlRI modelRI);
size_t SkUtGetModelDestroyCallbackCount();

void SkUtSetModelStreamNum(uint32_t streamNum);
uint32_t SkUtGetModelStreamNum();
void SkUtSetStreamTaskNum(uint32_t streamIdx, uint32_t taskNum);
uint32_t SkUtGetStreamTaskNum(uint32_t streamIdx);
void SkUtSetTaskType(uint32_t streamIdx, uint32_t taskIdx, aclrtTaskType type);
aclrtTaskType SkUtGetTaskType(uint32_t streamIdx, uint32_t taskIdx);

void SkUtSetEntryBinHandleNull(int enable);
void SkUtSetBinaryGetFunctionNullHandle(int enable);
void SkUtSetLastBinaryGetFunctionName(const char *funcName);

void SkUtSetSecurecMemcpyFailOnCall(int hitOnCall);
void SkUtSetSecurecMemsetFailOnCall(int hitOnCall);
int SkUtSecurecShouldFailMemcpy();
int SkUtSecurecShouldFailMemset();

void SkUtSetStreamId(uint32_t streamIdx, int32_t streamId);
int32_t SkUtGetStreamId(uint32_t streamIdx);

void SkUtSetAclrtGetSocName(const char *socName);
const char *SkUtGetAclrtGetSocName();

uint32_t SkUtGetDebugJsonPrintCallCount();
const char *SkUtGetDebugJsonPrintPath(uint32_t index);
void SkUtRecordDebugJsonPrintPath(const char *path);

#ifdef __cplusplus
}
#endif

#endif  // SUPER_KERNEL_UT_COMMON_STUBS_H_
