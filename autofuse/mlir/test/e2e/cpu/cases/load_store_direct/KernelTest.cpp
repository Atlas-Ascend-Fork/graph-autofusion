/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_store_direct/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_store_direct::GetShapeConfig;
using mlir::afir::load_store_direct::kInputShapes;
using mlir::afir::load_store_direct::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadStoreDirectCopiesInputToOutput) {
  const uint32_t input0_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(input0_element_count * sizeof(float)));
  // Keep the kernel pointer 32B aligned, with guards immediately outside the logical output.
  constexpr uint32_t kGuardElements = 8;
  constexpr float kCanary = -123456.0F;
  float *output_storage =
      static_cast<float *>(AscendC::GmAlloc((output_element_count + 2 * kGuardElements) * sizeof(float)));
  float *output = output_storage == nullptr ? nullptr : output_storage + kGuardElements;
  if (input0 == nullptr || output == nullptr) {
    if (input0 != nullptr) {
      AscendC::GmFree(input0);
    }
    if (output != nullptr) {
      AscendC::GmFree(output_storage);
    }
    FAIL() << "input/output allocation failed";
  }

  for (uint32_t index = 0; index < input0_element_count; ++index) {
    input0[index] = static_cast<float>(index) * 0.25F - 3.0F;
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  for (uint32_t index = 0; index < kGuardElements; ++index) {
    output_storage[index] = kCanary;
    output[output_element_count + index] = kCanary;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(input0);
    AscendC::GmFree(output_storage);
    FAIL() << "ATT tiling failed";
  }
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(input0);
      AscendC::GmFree(output_storage);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input0),
              reinterpret_cast<uint8_t *>(output), reinterpret_cast<uint8_t *>(workspace), tilingData);

  // 纯搬运：输出逐元素等于输入。
  for (uint32_t index = 0; index < output_element_count; ++index) {
    EXPECT_FLOAT_EQ(output[index], input0[index]) << "index=" << index;
  }

  for (uint32_t index = 0; index < kGuardElements; ++index) {
    EXPECT_FLOAT_EQ(output_storage[index], kCanary) << "prefix guard=" << index;
    EXPECT_FLOAT_EQ(output[output_element_count + index], kCanary) << "suffix guard=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output_storage);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
