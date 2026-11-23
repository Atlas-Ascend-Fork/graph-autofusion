/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <dlfcn.h>

namespace {
thread_local const char *g_profiler_stub_path = nullptr;
}

void SetPgoProfilerStubPath(const char *path) {
  g_profiler_stub_path = path;
}

extern "C" void *__real_dlopen(const char *path, int mode);

extern "C" void *__wrap_dlopen(const char *path, int mode) {
  if (g_profiler_stub_path != nullptr && path != nullptr &&
      (std::strcmp(path, "libprofapi.so") == 0 || std::strcmp(path, "libascend_hal.so") == 0)) {
    // Keep the stub local to calls made by the test codegen library.
    return __real_dlopen(g_profiler_stub_path, mode);
  }
  return __real_dlopen(path, mode);
}
