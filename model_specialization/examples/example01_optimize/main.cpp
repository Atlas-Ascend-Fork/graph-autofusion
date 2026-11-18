/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclmodel_specialization.h"
#include "../common/sample_resources.h"
#include <cmath>

// Check each call once; resources are released on normal completion or early return.
#define ASCENDC_CHECK(expr)                                                       \
  do {                                                                            \
    const auto checkRet = (expr);                                                 \
    if (checkRet != 0) {                                                          \
      std::fprintf(stderr, "%s failed: %d\n", #expr, static_cast<int>(checkRet)); \
      return 1;                                                                   \
    }                                                                             \
  } while (false)

bool PrepareInputs(SampleResources &resources) {
  std::array<float, kElementCount> x{}, y{};
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>(i);
    y[i] = 1.0F;
  }
  aclError ret = aclrtMemcpy(resources.buffers[0], kInputBytes, x.data(), kInputBytes, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "aclrtMemcpy(x) failed: %d\n", ret);
    return false;
  }
  ret = aclrtMemcpy(resources.buffers[1], kInputBytes, y.data(), kInputBytes, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "aclrtMemcpy(y) failed: %d\n", ret);
    return false;
  }
  return true;
}

bool VerifyResult(const std::array<float, kElementCount> &output, float increment) {
  for (size_t i = 0; i < output.size(); ++i) {
    const float expected = static_cast<float>(i) + increment;
    if (!std::isfinite(output[i]) || std::fabs(output[i] - expected) > 1.0e-6F) {
      std::fprintf(stderr, "Mismatch at %zu: output=%f expected=%f\n", i, output[i], expected);
      return false;
    }
  }
  std::printf("Numerical check passed: %zu float elements\n", output.size());
  return true;
}

int main(int argc, char **argv) {
  int32_t device = 0;
  if (argc > 2 || (argc == 2 && !ParseDevice(argv[1], device))) {
    std::fprintf(stderr, "Usage: demo [device_id]\n");
    return 1;
  }
  SampleResources resources;
  std::array<float, kElementCount> output{};
  if (!resources.Initialize(device) || !PrepareInputs(resources)) {
    return 1;
  }

  // Tensor order: x, y, t, z.
  ASCENDC_CHECK(aclnnAddCustomGetWorkspaceSize(resources.tensors[0], resources.tensors[1], resources.tensors[2],
                                               &resources.workspaceSizes[0], &resources.executors[0]));
  if (!resources.PrepareExecutor(0)) {
    return 1;
  }
  ASCENDC_CHECK(aclnnAddCustomGetWorkspaceSize(resources.tensors[2], resources.tensors[1], resources.tensors[3],
                                               &resources.workspaceSizes[1], &resources.executors[1]));
  if (!resources.PrepareExecutor(1)) {
    return 1;
  }

  ASCENDC_CHECK(aclmdlRICaptureBegin(resources.stream, ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL));
  resources.capturing = true;
  // Optimize covers both Add calls: t = x + y, then z = t + y.
  ASCENDC_CHECK(
      aclnnAddCustom(resources.workspaces[0], resources.workspaceSizes[0], resources.executors[0], resources.stream));
  ASCENDC_CHECK(
      aclnnAddCustom(resources.workspaces[1], resources.workspaceSizes[1], resources.executors[1], resources.stream));
  ASCENDC_CHECK(aclmdlRICaptureEnd(resources.stream, &resources.model));
  resources.capturing = false;

  ASCENDC_CHECK(aclmdlRISpecOptimize(resources.model, nullptr));
  ASCENDC_CHECK(aclmdlRIExecuteAsync(resources.model, resources.stream));
  ASCENDC_CHECK(aclrtSynchronizeStream(resources.stream));
  for (size_t i = 2; i < resources.buffers.size(); ++i) {
    ASCENDC_CHECK(
        aclrtMemcpy(output.data(), kOutputBytes, resources.buffers[i], kOutputBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!VerifyResult(output, static_cast<float>(i - 1))) {
      return 1;
    }
  }
  if (!resources.Close()) {
    std::fprintf(stderr, "Resource cleanup failed\n");
    return 1;
  }
  std::printf("Example passed\n");
  return 0;
}
