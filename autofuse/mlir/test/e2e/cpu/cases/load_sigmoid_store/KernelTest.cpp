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
#include "test/e2e/cpu/cases/load_sigmoid_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_sigmoid_store::GetShapeConfig;
using mlir::afir::load_sigmoid_store::kInputShapes;
using mlir::afir::load_sigmoid_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadSigmoidStoreCalculatesCorrectResult) {
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

  // 取值铺开到 ±4：sigmoid 在 0 附近近似线性、两端饱和到 0 / 1，一份只在中间对的实现
  // （比如原样输出、或者算成了 tanh）在两端就分得出来了。
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
    const float x = input[index];
    // 设备上那一份是近似实现，与精确式在 |x| < 4 上差不到 1e-3。1e-2 的容差容得下近似，
    // 容不下「算成了别的函数」。
    const float expected = 1.0F / (1.0F + std::exp(-x));
    EXPECT_NEAR(output[index], expected, 1e-2F) << "index=" << index;
    // sigmoid 恒落在 (0, 1) 里，且单调递增——写成 identity / relu / tanh 都会破这两条。
    EXPECT_GT(output[index], -1e-4F) << "index=" << index;
    EXPECT_LT(output[index], 1.0F + 1e-4F) << "index=" << index;
    if (index > 0U && input[index] > input[index - 1U]) {
      EXPECT_GE(output[index], output[index - 1U] - 1e-4F) << "index=" << index;
    }
  }

  AscendC::GmFree(input);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
