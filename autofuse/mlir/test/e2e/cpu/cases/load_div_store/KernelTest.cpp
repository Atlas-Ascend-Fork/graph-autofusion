/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_div_store/LoadDivStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_div_store::GetShapeConfig;
using mlir::afir::load_div_store::kInputShapes;
using mlir::afir::load_div_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadDivStoreCalculatesCorrectResult) {
  const uint32_t input0_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t input1_element_count = GetShapeConfig(kInputShapes, 1).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(input0_element_count * sizeof(float)));
  float *input1 = static_cast<float *>(AscendC::GmAlloc(input1_element_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  if (input0 == nullptr || input1 == nullptr || output == nullptr) {
    if (input0 != nullptr) {
      AscendC::GmFree(input0);
    }
    if (input1 != nullptr) {
      AscendC::GmFree(input1);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  // 除数取 ±[1, 4]，避免接近 0 时放大 CPU 模拟器与主机除法的舍入差异。
  for (uint32_t index = 0; index < input1_element_count; ++index) {
    const float magnitude = static_cast<float>((index % 4U) + 1U);
    input1[index] = (index % 2U == 0U) ? magnitude : -magnitude;
  }
  // 选择可精确表示的被除数和除数，使商为非零的 ±(k + 0.5)。
  for (uint32_t index = 0; index < input0_element_count; ++index) {
    const float quotient = static_cast<float>(static_cast<int32_t>(index % 5U) - 2) + 0.5F;
    input0[index] = input1[index] * quotient;
  }

  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(input0);
    AscendC::GmFree(input1);
    AscendC::GmFree(output);
    FAIL() << "ATT tiling failed";
  }
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(input0);
      AscendC::GmFree(input1);
      AscendC::GmFree(output);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input0),
              reinterpret_cast<uint8_t *>(input1), reinterpret_cast<uint8_t *>(output),
              reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    const float expected = input0[index] / input1[index];
    EXPECT_NEAR(output[index], expected, 1e-5F) << "index=" << index;
    // 期望商非零，避免错误调用因恰好得到零结果而通过。
    EXPECT_NE(output[index], 0.0F) << "index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
