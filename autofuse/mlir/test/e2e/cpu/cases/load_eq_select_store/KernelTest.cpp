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
#include "test/e2e/cpu/cases/load_eq_select_store/LoadEqSelectStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_eq_select_store::GetShapeConfig;
using mlir::afir::load_eq_select_store::kInputShapes;
using mlir::afir::load_eq_select_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2,
                                                                    GM_ADDR input3, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadEqSelectStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *lhs = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *rhs = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *onTrue = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *onFalse = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  if (lhs == nullptr || rhs == nullptr || onTrue == nullptr || onFalse == nullptr || output == nullptr) {
    FAIL() << "input/output allocation failed";
  }

  for (uint32_t index = 0; index < input_element_count; ++index) {
    lhs[index] = static_cast<float>(index % 5U) - 2.0F;
    rhs[index] = ((index % 2U) == 0U) ? lhs[index] : (lhs[index] + 1.0F);
    onTrue[index] = 100.0F + static_cast<float>(index);
    onFalse[index] = -100.0F - static_cast<float>(index);
  }
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
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(lhs),
              reinterpret_cast<uint8_t *>(rhs), reinterpret_cast<uint8_t *>(onTrue),
              reinterpret_cast<uint8_t *>(onFalse), reinterpret_cast<uint8_t *>(output),
              reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    const float expected = (lhs[index] == rhs[index]) ? onTrue[index] : onFalse[index];
    EXPECT_FLOAT_EQ(output[index], expected) << "index=" << index;
  }

  AscendC::GmFree(lhs);
  AscendC::GmFree(rhs);
  AscendC::GmFree(onTrue);
  AscendC::GmFree(onFalse);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
