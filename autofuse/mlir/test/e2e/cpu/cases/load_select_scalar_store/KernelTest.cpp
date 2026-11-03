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
#include <limits>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_select_scalar_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_select_scalar_store::GetShapeConfig;
using mlir::afir::load_select_scalar_store::kInputShapes;
using mlir::afir::load_select_scalar_store::kOutputShapes;

// 与建图时给的 Scalar 节点一致。它是 **then** 那一路，掩码为真时取它。
constexpr float kOnTrue = 3.0F;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadSelectScalarStoreCalculatesCorrectResult) {
  const uint32_t probe_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t value_count = GetShapeConfig(kInputShapes, 1).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *input0 = static_cast<float *>(AscendC::GmAlloc(probe_count * sizeof(float)));
  auto *input1 = static_cast<float *>(AscendC::GmAlloc(value_count * sizeof(float)));
  auto *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(output, nullptr);

  // 掩码由 `Isnan(input0)` 现算，所以 input0 要 NaN 与非 NaN 混着来；两个取值都出现，
  // 否则「恒取标量」或「恒取 tile」也能过。
  //
  // 值那一路取 0.25 + 0.5k，**永远取不到 3.0**：两路的实参位置换过来算的是
  // `isnan(p) ? v : 3.0`，与这里的期望在每一个元素上都不同。取值在 f32 里精确，
  // 而且 Where 只搬位不做算术，所以逐元素比相等。
  for (uint32_t index = 0; index < probe_count; ++index) {
    input0[index] = index % 3U == 0U ? std::numeric_limits<float>::quiet_NaN() : 0.0F;
  }
  for (uint32_t index = 0; index < value_count; ++index) {
    input1[index] = static_cast<float>(static_cast<int32_t>(index % 9U) - 4) * 0.5F + 0.25F;
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
              reinterpret_cast<GM_ADDR>(input1), reinterpret_cast<GM_ADDR>(output),
              reinterpret_cast<GM_ADDR>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    // **掩码为真取标量**。两路的实参位置换过来这里逐元素红。
    const float expected = std::isnan(input0[index]) ? kOnTrue : input1[index];
    EXPECT_FLOAT_EQ(output[index], expected) << "element " << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
