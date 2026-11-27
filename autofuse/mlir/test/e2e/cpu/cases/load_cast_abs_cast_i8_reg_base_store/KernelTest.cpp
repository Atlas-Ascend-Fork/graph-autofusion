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
#include "test/e2e/cpu/cases/load_cast_abs_cast_i8_reg_base_store/LoadCastAbsCastI8RegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_cast_abs_cast_i8_reg_base_store::GetShapeConfig;
using mlir::afir::load_cast_abs_cast_i8_reg_base_store::kInputShapes;
using mlir::afir::load_cast_abs_cast_i8_reg_base_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadCastAbsCastI8RegBaseStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *input0 = static_cast<int8_t *>(AscendC::GmAlloc(input_element_count * sizeof(int8_t)));
  auto *output = static_cast<int8_t *>(AscendC::GmAlloc(output_element_count * sizeof(int8_t)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // 取值落在 [-127, 127] 且不为 0，整数在 half 里到 2048 都精确，所以中间那一趟不引入
  // 误差，回来时也碰不到饱和——下面可以逐元素比相等。不取 -128 的理由同
  // `load_abs_neg_i8_reg_base_store`。一半取负：`abs` 在全正的输入上退化成恒等变换。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    const int32_t magnitude = static_cast<int32_t>(index % 127U) + 1;
    input0[index] = static_cast<int8_t>((index % 2U == 0U) ? magnitude : -magnitude);
  }
  // 哨兵不是 0，也不是任何一点输入的绝对值（绝对值都是正的）。
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = static_cast<int8_t>(-1);
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
    const auto expected = static_cast<int8_t>(input0[index] < 0 ? -static_cast<int32_t>(input0[index])
                                                                : static_cast<int32_t>(input0[index]));
    EXPECT_EQ(output[index], expected) << "element " << index << " input=" << static_cast<int32_t>(input0[index]);
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
