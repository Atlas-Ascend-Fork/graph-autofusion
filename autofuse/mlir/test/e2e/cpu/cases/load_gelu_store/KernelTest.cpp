/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_gelu_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_gelu_store::GetShapeConfig;
using mlir::afir::load_gelu_store::kInputShapes;
using mlir::afir::load_gelu_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadGeluStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  if (input == nullptr || output == nullptr) {
    if (input != nullptr) {
      AscendC::GmFree(input);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  // 取值铺开到 ±4:GELU 在 0 附近弯曲最厉害，在两端分别贴近 0 和 x，一份只在中间对的
  // 实现分不出来。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input[index] = (static_cast<float>(index % 17U) - 8.0F) * 0.5F;
  }

  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(input);
    AscendC::GmFree(output);
    FAIL() << "ATT tiling failed";
  }
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(input);
      AscendC::GmFree(output);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input),
              reinterpret_cast<uint8_t *>(output), reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    // 精确式 `0.5x(1 + erf(x/sqrt(2)))`。设备上那一份可能是 tanh 近似，两者在 |x| < 4
    // 上差不到 1e-3，所以 1e-2 的容差既容得下近似，又容不下「算成了别的函数」。
    const float x = input[index];
    const float expected = 0.5F * x * (1.0F + std::erf(x / std::sqrt(2.0F)));
    EXPECT_NEAR(output[index], expected, 1e-2F) << "index=" << index;
    // GELU 恒被夹在 0 与 x 之间（含端点）。写成 relu / identity / sigmoid 都会破这条。
    EXPECT_LE(output[index] * (x >= 0.0F ? 1.0F : -1.0F), std::fabs(x) + 1e-4F) << "index=" << index;
  }

  AscendC::GmFree(input);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
