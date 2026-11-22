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
#include "test/e2e/cpu/cases/load_remainder_store/LoadRemainderStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_remainder_store::GetShapeConfig;
using mlir::afir::load_remainder_store::kInputShapes;
using mlir::afir::load_remainder_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadRemainderStoreCalculatesCorrectResult) {
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

  // 包装算的是 `dividend - floor(dividend / divisor) * divisor`——符号跟除数走，
  // 与 C 的 `fmod`（符号跟被除数走）在混合符号下答案不同。用例要能把这两者分开，所以
  // 除数正负都有，商也正负都有。
  //
  // 构造法：除数取小整数 d，商取 q ∈ {-1.5, -0.5, 0.5, 1.5, 2.5}，被除数 = d * q。
  // 三者在 float 里都精确，而且商离整数边界恒差 0.5——设备上的 Div 与主机的 `/`
  // 差一个 ulp 也不会让 floor 跳到另一个整数去，用例不会偶发。
  // 于是期望值恒为 `d * 0.5`，符号就是 d 的符号。
  for (uint32_t index = 0; index < input1_element_count; ++index) {
    const float magnitude = static_cast<float>((index % 7U) + 2U);
    input1[index] = (index % 2U == 0U) ? magnitude : -magnitude;
  }
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
    const float expected = input0[index] - std::floor(input0[index] / input1[index]) * input1[index];
    EXPECT_NEAR(output[index], expected, 1e-4F) << "index=" << index;
    // 上面那条要是被一份「按 fmod 实现」的包装骗过去，这条会拦住：余数与除数同号。
    EXPECT_GT(output[index] * input1[index], 0.0F) << "index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
