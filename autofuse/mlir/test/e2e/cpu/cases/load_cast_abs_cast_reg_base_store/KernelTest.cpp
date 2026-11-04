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
#include "test/e2e/cpu/cases/load_cast_abs_cast_reg_base_store/LoadCastAbsCastRegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_cast_abs_cast_reg_base_store::GetShapeConfig;
using mlir::afir::load_cast_abs_cast_reg_base_store::kInputShapes;
using mlir::afir::load_cast_abs_cast_reg_base_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadCastAbsCastRegBaseStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *input0 = static_cast<half *>(AscendC::GmAlloc(input_element_count * sizeof(half)));
  auto *output = static_cast<half *>(AscendC::GmAlloc(output_element_count * sizeof(half)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // 取值全部在 half 里精确可表示（[-13, 13] 的半整数，half 的尾数 10 位），所以中间那趟
  // f32 不引入误差，下面可以逐元素比相等而不是比近似——近似比对会把「少转了一次」这种错
  // 放过去。
  //
  // **一半取负**：`abs` 在全正的输入上退化成恒等变换，那样漏发 `Abs` 看不出来。
  // **不取 0**：输出哨兵与「算出 0」要分得开。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    const float magnitude = static_cast<float>(index % 13U) + 0.5F;
    input0[index] = half((index % 2U == 0U) ? magnitude : -magnitude);
  }
  // 哨兵而不是 0。
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = half(-12345.0F);
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
    const float expected = std::abs(static_cast<float>(input0[index]));
    EXPECT_EQ(static_cast<float>(output[index]), expected) << "element " << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
