/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/workspace_readback/WorkspaceReadbackCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::workspace_readback::GetShapeConfig;
using mlir::afir::workspace_readback::kInputShapes;
using mlir::afir::workspace_readback::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR output2,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, WorkspaceReadbackRoundTripsThroughGm) {
  const uint32_t input0_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(input0_element_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  float *output2 = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  if (input0 == nullptr || output == nullptr || output2 == nullptr) {
    if (input0 != nullptr) {
      AscendC::GmFree(input0);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    if (output2 != nullptr) {
      AscendC::GmFree(output2);
    }
    FAIL() << "input/output allocation failed";
  }

  for (uint32_t index = 0; index < input0_element_count; ++index) {
    input0[index] = static_cast<float>(index) * 0.25F - 3.0F;
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
    output2[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = UINT64_MAX;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(input0);
    AscendC::GmFree(output);
    AscendC::GmFree(output2);
    FAIL() << "ATT tiling failed";
  }
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(input0);
      AscendC::GmFree(output);
      AscendC::GmFree(output2);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input0),
              reinterpret_cast<uint8_t *>(output), reinterpret_cast<uint8_t *>(output2),
              reinterpret_cast<uint8_t *>(workspace), tilingData);

  // 第一路写出 2x，第二路读回并翻倍得到 4x。
  // 检查第二路结果，以发现回读早于写入完成导致的初始值或错误数据。
  for (uint32_t index = 0; index < output_element_count; ++index) {
    EXPECT_FLOAT_EQ(output[index], 2.0F * input0[index]) << "index=" << index;
    EXPECT_FLOAT_EQ(output2[index], 4.0F * input0[index]) << "index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(output2);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
