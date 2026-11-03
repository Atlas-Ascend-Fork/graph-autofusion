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
#include "test/e2e/cpu/cases/load_scalar_div_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_scalar_div_store::GetShapeConfig;
using mlir::afir::load_scalar_div_store::kInputShapes;
using mlir::afir::load_scalar_div_store::kOutputShapes;

// 与建图时给的 Scalar 节点一致。
constexpr float kDividend = 4.0F;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadScalarDivStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *input0 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // 取值是 ±2^k（k 取 -2..1），除数与商在 f32 里都精确，所以逐元素比相等。**不能有 0**：
  // 被除数是常量 4.0，除以 0 出来的是 inf，两边的 inf 比相等能过，钉不住任何东西。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    const float magnitude = 0.25F * static_cast<float>(1U << (index % 4U));
    input0[index] = (index % 8U) < 4U ? magnitude : -magnitude;
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  auto *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0 ? 1 : workspaceSize));
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<GM_ADDR>(input0),
              reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    // **`4.0F / x`，不是 `x / 4.0F`**。方向那一位写反的话这里逐元素红——两者相等要
    // |x| = 4，而取值全在 [0.25, 2] 里，一个都不是。
    EXPECT_FLOAT_EQ(output[index], kDividend / input0[index]) << "element " << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
