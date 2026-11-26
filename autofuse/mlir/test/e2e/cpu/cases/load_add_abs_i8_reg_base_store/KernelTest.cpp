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
#include "test/e2e/cpu/cases/load_add_abs_i8_reg_base_store/LoadAddAbsI8RegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_add_abs_i8_reg_base_store::GetShapeConfig;
using mlir::afir::load_add_abs_i8_reg_base_store::kInputShapes;
using mlir::afir::load_add_abs_i8_reg_base_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadAddAbsI8RegBaseStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *lhs = static_cast<int8_t *>(AscendC::GmAlloc(input_element_count * sizeof(int8_t)));
  auto *rhs = static_cast<int8_t *>(AscendC::GmAlloc(input_element_count * sizeof(int8_t)));
  auto *output = static_cast<int8_t *>(AscendC::GmAlloc(output_element_count * sizeof(int8_t)));
  ASSERT_NE(lhs, nullptr);
  ASSERT_NE(rhs, nullptr);
  ASSERT_NE(output, nullptr);

  // 两个加数与它们的和都落在 [-127, 127] 内，和不为 0：不碰 -128 与溢出——溢出后是饱和还是
  // 回绕由指令的模式定，那不是这一格要钉的东西。和有正有负，`abs` 才不是恒等变换。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    const int32_t a = static_cast<int32_t>(index % 61U) - 30;  // [-30, 30]
    const int32_t b = ((index % 3U) == 0U) ? 40 : -45;
    lhs[index] = static_cast<int8_t>(a);
    rhs[index] = static_cast<int8_t>(b);
  }
  // 哨兵是个负数：结果是绝对值，恒非负。
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
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(lhs),
              reinterpret_cast<uint8_t *>(rhs), reinterpret_cast<uint8_t *>(output), workspace, tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    // 取值范围内每一步都精确，所以用等号不用容差。
    const int32_t sum = static_cast<int32_t>(lhs[index]) + static_cast<int32_t>(rhs[index]);
    const auto expected = static_cast<int8_t>(sum < 0 ? -sum : sum);
    EXPECT_EQ(output[index], expected) << "element " << index << " lhs=" << static_cast<int32_t>(lhs[index])
                                       << " rhs=" << static_cast<int32_t>(rhs[index]);
  }

  AscendC::GmFree(lhs);
  AscendC::GmFree(rhs);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
