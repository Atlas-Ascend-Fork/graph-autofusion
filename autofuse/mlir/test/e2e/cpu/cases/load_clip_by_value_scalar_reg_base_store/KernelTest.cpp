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
#include "test/e2e/cpu/cases/load_clip_by_value_scalar_reg_base_store/LoadClipByValueScalarRegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_clip_by_value_scalar_reg_base_store::GetShapeConfig;
using mlir::afir::load_clip_by_value_scalar_reg_base_store::kInputShapes;
using mlir::afir::load_clip_by_value_scalar_reg_base_store::kOutputShapes;

// 与建图时给的两个 Scalar 节点一致。不对称：互换就是另一个函数。
constexpr float kLower = -1.0F;
constexpr float kUpper = 1.0F;
}  // namespace

// min / max 不是 kernel 输入——它们是编译期常量，落在 kernel 顶部的 `constexpr`。
extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadClipByValueScalarRegBaseStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *input0 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // 取值两侧都超界，而且不对称：`[-2, 1.75]` 的 0.25 步长。只截一侧的错、把上下界互换的错
  // 都会在这份取值上露出来。全部在 f32 里精确可表示，所以逐元素比相等。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = static_cast<float>(static_cast<int32_t>(index % 16U) - 8) * 0.25F;
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
    const float expected = input0[index] < kLower ? kLower : (input0[index] > kUpper ? kUpper : input0[index]);
    EXPECT_FLOAT_EQ(output[index], expected) << "element " << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
