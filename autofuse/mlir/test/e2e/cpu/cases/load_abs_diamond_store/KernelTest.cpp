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
#include <vector>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_abs_diamond_store/LoadAbsDiamondStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_abs_diamond_store::GetShapeConfig;
using mlir::afir::load_abs_diamond_store::kInputShapes;
using mlir::afir::load_abs_diamond_store::kOutputShapes;
constexpr uint32_t kInputCount = 3U;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2,
                                                                    GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadAbsDiamondStoreCalculatesCorrectResult) {
  std::vector<float *> inputs(kInputCount, nullptr);
  for (uint32_t slot = 0U; slot < kInputCount; ++slot) {
    const uint32_t count = GetShapeConfig(kInputShapes, slot).element_count;
    inputs[slot] = static_cast<float *>(AscendC::GmAlloc(count * sizeof(float)));
    ASSERT_NE(inputs[slot], nullptr) << "input allocation failed: slot=" << slot;
    // slot 0 同时包含正负值，负输入用于发现遗漏 Abs。
    for (uint32_t index = 0; index < count; ++index) {
      inputs[slot][index] = static_cast<float>(index) * 0.25F - static_cast<float>(slot) - 3.0F;
    }
  }

  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  ASSERT_NE(output, nullptr) << "output allocation failed";
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0) << "ATT tiling failed";
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    ASSERT_NE(workspace, nullptr) << "workspace allocation failed";
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(inputs[0]),
              reinterpret_cast<uint8_t *>(inputs[1]), reinterpret_cast<uint8_t *>(inputs[2]),
              reinterpret_cast<uint8_t *>(output), reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    const float fork = inputs[0][index] < 0.0F ? -inputs[0][index] : inputs[0][index];
    const float expected = (fork + inputs[1][index]) + (fork + inputs[2][index]);
    EXPECT_FLOAT_EQ(output[index], expected) << "index=" << index;
  }

  for (float *input : inputs) {
    AscendC::GmFree(input);
  }
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
