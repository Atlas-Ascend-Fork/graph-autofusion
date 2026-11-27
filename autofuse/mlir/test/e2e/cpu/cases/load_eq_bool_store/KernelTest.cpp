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
#include "test/e2e/cpu/cases/load_eq_bool_store/LoadEqBoolStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_eq_bool_store::GetShapeConfig;
using mlir::afir::load_eq_bool_store::kInputShapes;
using mlir::afir::load_eq_bool_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadEqBoolStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  // 输入 f32、输出一个字节的真值，两边按各自的元素宽度分配。
  auto *lhs = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *rhs = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *output = static_cast<uint8_t *>(AscendC::GmAlloc(output_element_count * sizeof(uint8_t)));
  ASSERT_NE(lhs, nullptr);
  ASSERT_NE(rhs, nullptr);
  ASSERT_NE(output, nullptr);

  // 取值与 `load_eq_store` 同一套：相等与不相等各占一半，且不相等的那一半两个方向都有。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    lhs[index] = static_cast<float>(index % 5U) - 2.0F;
    rhs[index] = ((index % 2U) == 0U) ? lhs[index] : (lhs[index] + static_cast<float>((index % 4U) == 1U ? 1 : -1));
  }
  // 哨兵既不是 0 也不是 1：结果只有这两种取值，「kernel 一个字都没写」要能和「算出 0」
  // 分得开。
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0xA5U;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  auto *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0U ? 1U : workspaceSize));
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(lhs),
              reinterpret_cast<uint8_t *>(rhs), output, workspace, tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    // 结果只有 0 / 1，所以用等号不用容差。
    const auto expected = static_cast<uint8_t>(lhs[index] == rhs[index] ? 1U : 0U);
    EXPECT_EQ(output[index], expected) << "index=" << index << " lhs=" << lhs[index] << " rhs=" << rhs[index];
  }

  AscendC::GmFree(lhs);
  AscendC::GmFree(rhs);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
