/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_eq_i8_reg_base_store/LoadEqI8RegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_eq_i8_reg_base_store::GetShapeConfig;
using mlir::afir::load_eq_i8_reg_base_store::kInputShapes;
using mlir::afir::load_eq_i8_reg_base_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadEqI8RegBaseStoreCalculatesCorrectResult) {
  const auto release = [](void *buffer) { AscendC::GmFree(buffer); };
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  std::unique_ptr<int8_t[], decltype(release)> input0(
      static_cast<int8_t *>(AscendC::GmAlloc(input_element_count * sizeof(int8_t))), release);
  std::unique_ptr<int8_t[], decltype(release)> input1(
      static_cast<int8_t *>(AscendC::GmAlloc(input_element_count * sizeof(int8_t))), release);
  std::unique_ptr<uint8_t[], decltype(release)> output(
      static_cast<uint8_t *>(AscendC::GmAlloc(output_element_count * sizeof(uint8_t))), release);
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(output, nullptr);

  // 值域覆盖负数与 INT8_MIN；两边同源再按步进错开一部分，相等/不等都有成片样本。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = static_cast<int8_t>(static_cast<int32_t>(index % 7U) - 3);
    input1[index] =
        static_cast<int8_t>(static_cast<int32_t>(index % 7U) - 3 + ((index % 11U == 5U || index % 97U == 0U) ? 1 : 0));
    if (index % 97U == 0U) {
      input0[index] = INT8_MIN;
      input1[index] = INT8_MIN;
    }
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0x5AU;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  std::unique_ptr<uint8_t[], decltype(release)> workspace(
      static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0 ? 1 : workspaceSize)), release);
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<GM_ADDR>(input0.get()),
              reinterpret_cast<GM_ADDR>(input1.get()), reinterpret_cast<GM_ADDR>(output.get()),
              reinterpret_cast<GM_ADDR>(workspace.get()), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    const uint8_t expected = (input0[index] == input1[index]) ? 1U : 0U;
    EXPECT_EQ(output[index], expected) << "element " << index;
  }
}
