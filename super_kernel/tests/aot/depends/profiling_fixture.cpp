/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "profiling_fixture.h"
#include "aprof_pub.h"
namespace {
std::string profilingPath = "/tmp/prof_output/mindstudio_profiler_output";
int32_t (*profilingCallback)(uint32_t, void *, uint32_t) = nullptr;
uint32_t profilingType = 0;
}  // namespace
char *MsprofGetPath() {
  return profilingPath.data();
}
int32_t MsprofRegisterCallback(uint32_t type, int32_t (*callback)(uint32_t, void *, uint32_t)) {
  profilingType = type;
  profilingCallback = callback;
  return 0;
}
namespace sk::test {
void SetProfilingPath(const std::string &path) {
  profilingPath = path;
}
bool HasProfilingCallback() {
  return profilingCallback != nullptr;
}
bool NotifyProfiling() {
  return profilingCallback != nullptr && profilingCallback(profilingType, nullptr, 0) == 0;
}
}  // namespace sk::test
