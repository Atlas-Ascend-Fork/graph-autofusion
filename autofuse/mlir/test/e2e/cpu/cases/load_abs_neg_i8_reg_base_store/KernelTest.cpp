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
#include "test/e2e/cpu/cases/load_abs_neg_i8_reg_base_store/LoadAbsNegI8RegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_abs_neg_i8_reg_base_store::GetShapeConfig;
using mlir::afir::load_abs_neg_i8_reg_base_store::kInputShapes;
using mlir::afir::load_abs_neg_i8_reg_base_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadAbsNegI8RegBaseStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *input0 = static_cast<int8_t *>(AscendC::GmAlloc(input_element_count * sizeof(int8_t)));
  auto *output = static_cast<int8_t *>(AscendC::GmAlloc(output_element_count * sizeof(int8_t)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // 取值落在 [-127, 127] 且不为 0。不取 -128：它的绝对值 128 在 int8 里表示不了，溢出后
  // 是饱和还是回绕由指令的饱和模式定，那不是这一格要钉的东西。一半取负——`abs` 在全正的
  // 输入上退化成恒等变换，那样漏生成 `Abs` 看不出来。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    const int32_t magnitude = static_cast<int32_t>(index % 127U) + 1;
    input0[index] = static_cast<int8_t>((index % 2U == 0U) ? magnitude : -magnitude);
  }
  // 哨兵取 0：结果是绝对值再取负，恒为负，0 不会是任何一点的真值。
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = static_cast<int8_t>(0);
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  auto *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0U ? 1U : workspaceSize));
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input0),
              reinterpret_cast<uint8_t *>(output), workspace, tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    // 取值范围内每一步都精确，所以用等号不用容差。
    const int32_t magnitude =
        input0[index] < 0 ? -static_cast<int32_t>(input0[index]) : static_cast<int32_t>(input0[index]);
    const auto expected = static_cast<int8_t>(-magnitude);
    EXPECT_EQ(output[index], expected) << "element " << index << " input=" << static_cast<int32_t>(input0[index]);
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
