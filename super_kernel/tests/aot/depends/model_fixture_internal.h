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
#include "model_fixture.h"

namespace sk::test {
// Shared layout for owned tasks; legacy UT retain the same three-field prefix.
struct RuntimeTask {
  uint32_t task_id = 0;
  aclmdlRITaskType type = ACL_MODEL_RI_TASK_DEFAULT;
  aclmdlRITaskParams params{};
};
// These adapters are used only by the external API stubs. Unknown legacy UT
// handles are left to the existing stub behavior, without a global ST mode.
bool ModelStreams(aclmdlRI model, aclrtStream *streams, uint32_t *count, aclError &result);
bool StreamTasks(aclrtStream stream, aclmdlRITask *tasks, uint32_t *count, aclError &result);
bool StreamId(aclrtStream stream, int32_t &id);
void RecordUpdate(aclmdlRI model, aclError result);
bool ValidTaskUpdate(aclmdlRITask task, const aclmdlRITaskParams &params);
void RecordTaskUpdate(aclmdlRITask task, const aclmdlRITaskParams &params);
void RecordDisable(aclmdlRITask task);
void RecordLaunch(aclrtStream stream, const char *function, const char *scope, size_t size);
bool FunctionName(aclrtFuncHandle function, std::string &name);
bool FunctionAttribute(aclrtFuncHandle function, aclrtFuncAttribute attr, int64_t &value);
size_t BinaryMetadataCount(aclrtBinHandle binary);
aclrtFuncHandle ResolveFunction(const char *name);
bool FunctionBinary(aclrtFuncHandle function, aclrtBinHandle &binary);
bool FunctionAddress(aclrtFuncHandle function, void **cube, void **vector);
bool BinaryAddress(aclrtBinHandle binary, void **address, size_t *size);
bool BinaryMetadata(aclrtBinHandle binary, size_t count, void **data, size_t *sizes, int &result);
bool HasBinary(aclrtBinHandle binary);
void TrackAllocation(void *address, size_t size);
void ForgetAllocation(void *address);
void SetRegisteredMemory(void *address, int value, size_t count);
bool CopyRegisteredMemory(void *dst, const void *src, size_t count, aclrtMemcpyKind kind);
}  // namespace sk::test
