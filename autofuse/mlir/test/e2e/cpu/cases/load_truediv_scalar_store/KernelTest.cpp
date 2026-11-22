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
#include "test/e2e/cpu/cases/load_truediv_scalar_store/LoadTruedivScalarStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_truediv_scalar_store::GetShapeConfig;
using mlir::afir::load_truediv_scalar_store::kInputShapes;
using mlir::afir::load_truediv_scalar_store::kOutputShapes;

// 与建图时给的 Scalar 节点一致。i32 的除数，商是 f32。
constexpr int32_t kDivisor = 4;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadTrueDivScalarStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *input0 = static_cast<int32_t *>(AscendC::GmAlloc(input_element_count * sizeof(int32_t)));
  auto *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // 取 [-9, 9] 里的整数，除不尽的居多：除得尽的话整条 `Cast` 到 f32 再除的路与「整数
  // 相除」出来的结果一样，`truediv` 与 `floor_div` 就分不开了。商是四分之一的整数倍，
  // 在 f32 里精确，所以逐元素比相等。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = static_cast<int32_t>(index % 19U) - 9;
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
    // `x / 4`，不是 `4 / x`。方向那一位写反的话这里逐元素红——两者相等要 |x| = 2，
    // 而 19 个取值里只有 2 个是；x = 0 那一档更明显：0.0 对 inf。
    EXPECT_FLOAT_EQ(output[index], static_cast<float>(input0[index]) / static_cast<float>(kDivisor))
        << "element " << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
