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
#include "test/e2e/cpu/cases/load_mul_abs_reg_base_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_mul_abs_reg_base_store::GetShapeConfig;
using mlir::afir::load_mul_abs_reg_base_store::kInputShapes;
using mlir::afir::load_mul_abs_reg_base_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadMulAbsRegBaseStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  float *input1 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(output, nullptr);

  // 乘积要跨过 0：全正的乘积会让 `abs` 退化成恒等变换，那样「漏发 Abs」看不出来。
  // 两个输入都不取 0，免得「没算」被读成「算出 0」；也都不取 ±1，免得「漏发 Mul」碰巧对。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = (static_cast<float>(index % 23) + 2.0F) * 0.5F;
    input1[index] = ((index % 2) == 0 ? 1.0F : -1.0F) * (static_cast<float>(index % 37) + 2.0F) * 0.25F;
  }
  // 哨兵而不是 0：输出没被写时要能和「算出 0」分开。
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = -12345.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  uint8_t *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0 ? 1 : workspaceSize));
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<GM_ADDR>(input0),
              reinterpret_cast<GM_ADDR>(input1), reinterpret_cast<GM_ADDR>(output),
              reinterpret_cast<GM_ADDR>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    const float expected = std::abs(input0[index] * input1[index]);
    EXPECT_FLOAT_EQ(output[index], expected) << "element " << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
