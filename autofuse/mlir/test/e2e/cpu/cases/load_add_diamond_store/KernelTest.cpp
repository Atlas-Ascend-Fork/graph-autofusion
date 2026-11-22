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
#include "test/e2e/cpu/cases/load_add_diamond_store/LoadAddDiamondStoreCase.h"
#include "test/e2e/cpu/support/E2eTilingShape.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_add_diamond_store::GetShapeConfig;
using mlir::afir::load_add_diamond_store::kInputShapes;
using mlir::afir::load_add_diamond_store::kOutputShapes;
constexpr uint32_t kInputCount = 4U;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2,
                                                                    GM_ADDR input3, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadAddDiamondStoreCalculatesCorrectResult) {
  std::vector<float *> inputs(kInputCount, nullptr);
  for (uint32_t slot = 0U; slot < kInputCount; ++slot) {
    const uint32_t count = GetShapeConfig(kInputShapes, slot).element_count;
    inputs[slot] = static_cast<float *>(AscendC::GmAlloc(count * sizeof(float)));
    ASSERT_NE(inputs[slot], nullptr) << "input allocation failed: slot=" << slot;
    for (uint32_t index = 0; index < count; ++index) {
      inputs[slot][index] = static_cast<float>(index) * 0.25F - static_cast<float>(slot);
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

  // Check which scheduled-transfer offset terms this run exercises.
  // ATT chooses the split at run time; correct output alone does not prove the registered coverage.
  const mlir::afir::TilingShape tiling_shape = AF_E2E_TILING_SHAPE(tilingData, blockDim, output_element_count);
  mlir::afir::ReportTilingShape(AF_MLIR_E2E_CPU_CASE_NAME, tiling_shape);
  AF_E2E_ASSERT_TILING_COVERAGE(tiling_shape);
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    ASSERT_NE(workspace, nullptr) << "workspace allocation failed";
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(inputs[0]),
              reinterpret_cast<uint8_t *>(inputs[1]), reinterpret_cast<uint8_t *>(inputs[2]),
              reinterpret_cast<uint8_t *>(inputs[3]), reinterpret_cast<uint8_t *>(output),
              reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    const float add0 = inputs[0][index] + inputs[1][index];
    const float expected = (add0 + inputs[2][index]) + (add0 + inputs[3][index]);
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
