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
#include "test/e2e/cpu/cases/load_cast_reg_base_store/LoadCastRegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_cast_reg_base_store::GetShapeConfig;
using mlir::afir::load_cast_reg_base_store::kInputShapes;
using mlir::afir::load_cast_reg_base_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadCastRegBaseStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  int32_t *output = static_cast<int32_t *>(AscendC::GmAlloc(output_element_count * sizeof(int32_t)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // `f32 -> i32` 那一行的 round mode 是 `CAST_TRUNC`——向零截断，不是四舍五入。
  // 所以小数部分恒取 ±0.5：一份按 `lrintf` 写的实现会在这里全错，而按
  // `truncf` 写的全对，两者分得开。正负都要有，截断方向在负数上才看得出来。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    const float magnitude = static_cast<float>(index % 13U) + 0.5F;
    input0[index] = (index % 2U == 0U) ? magnitude : -magnitude;
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = -1;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  uint8_t *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0 ? 1 : workspaceSize));
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<GM_ADDR>(input0),
              reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    EXPECT_EQ(output[index], static_cast<int32_t>(std::trunc(input0[index]))) << "element " << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
