/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MODEL_SPECIALIZATION_SAMPLE_RESOURCES_H
#define MODEL_SPECIALIZATION_SAMPLE_RESOURCES_H
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_add_custom.h"
#include <array>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

constexpr int64_t kElementCount = 1024;
constexpr size_t kInputBytes = kElementCount * sizeof(float);
constexpr size_t kOutputBytes = kInputBytes;

inline bool ParseDevice(const char *text, int32_t &device) {
  if (text == nullptr) {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < 0 || value > INT32_MAX) {
    return false;
  }
  device = static_cast<int32_t>(value);
  return true;
}

// Own Runtime resources and release them on normal completion or early return.
struct SampleResources {
  bool initialized = false;
  bool capturing = false;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  aclmdlRI model = nullptr;
  std::array<void *, 4> buffers{};
  std::array<aclTensor *, 4> tensors{};
  std::array<aclOpExecutor *, 2> executors{};
  std::array<void *, 2> workspaces{};
  std::array<uint64_t, 2> workspaceSizes{};

  bool Initialize(int32_t device) {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "aclInit failed: %d\n", ret);
      return false;
    }
    initialized = true;
    ret = aclrtCreateContext(&context, device);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "aclrtCreateContext failed: %d\n", ret);
      return false;
    }
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
      return false;
    }
    const int64_t shape[] = {kElementCount};
    for (size_t i = 0; i < buffers.size(); ++i) {
      const size_t bytes = kInputBytes;
      ret = aclrtMalloc(&buffers[i], bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
      if (ret != ACL_SUCCESS || buffers[i] == nullptr) {
        return false;
      }
      if (i >= 2 && aclrtMemset(buffers[i], bytes, 0xFF, bytes) != ACL_SUCCESS) {
        return false;
      }
      tensors[i] = aclCreateTensor(shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 1, buffers[i]);
      if (tensors[i] == nullptr) {
        return false;
      }
    }
    return true;
  }

  bool PrepareExecutor(size_t index) {
    if (index >= executors.size() || executors[index] == nullptr) {
      return false;
    }
    // The captured model reuses these arguments; keep the executor alive until model destruction.
    aclError ret = aclSetAclOpExecutorRepeatable(executors[index]);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "aclSetAclOpExecutorRepeatable failed: %d\n", ret);
      return false;
    }
    if (workspaceSizes[index] != 0) {
      ret = aclrtMalloc(&workspaces[index], workspaceSizes[index], ACL_MEM_MALLOC_NORMAL_ONLY);
      if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtMalloc(workspace) failed: %d\n", ret);
        return false;
      }
    }
    return true;
  }

  bool Close() {
    if (capturing) {
      if (aclmdlRICaptureEnd(stream, &model) != ACL_SUCCESS) {
        return false;
      }
      capturing = false;
    }
    if (stream != nullptr && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
      return false;
    }
    if (model != nullptr) {
      if (aclmdlRIDestroy(model) != ACL_SUCCESS) {
        return false;
      }
      model = nullptr;
    }
    for (auto &executor : executors) {
      if (executor != nullptr) {
        if (aclDestroyAclOpExecutor(executor) != 0) {
          return false;
        }
        executor = nullptr;
      }
    }
    for (auto &tensor : tensors) {
      if (tensor != nullptr) {
        if (aclDestroyTensor(tensor) != 0) {
          return false;
        }
        tensor = nullptr;
      }
    }
    for (auto &buffer : buffers) {
      if (buffer != nullptr) {
        if (aclrtFree(buffer) != ACL_SUCCESS) {
          return false;
        }
        buffer = nullptr;
      }
    }
    for (auto &workspace : workspaces) {
      if (workspace != nullptr) {
        if (aclrtFree(workspace) != ACL_SUCCESS) {
          return false;
        }
        workspace = nullptr;
      }
    }
    if (stream != nullptr) {
      if (aclrtDestroyStream(stream) != ACL_SUCCESS) {
        return false;
      }
      stream = nullptr;
    }
    if (context != nullptr) {
      if (aclrtDestroyContext(context) != ACL_SUCCESS) {
        return false;
      }
      context = nullptr;
    }
    if (initialized) {
      if (aclFinalize() != ACL_SUCCESS) {
        return false;
      }
      initialized = false;
    }
    return true;
  }

  ~SampleResources() {
    if (!Close()) {
      std::fprintf(stderr, "Resource cleanup failed\n");
    }
  }
};
#endif
