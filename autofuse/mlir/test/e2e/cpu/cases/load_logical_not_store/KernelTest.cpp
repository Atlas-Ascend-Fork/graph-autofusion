/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_logical_not_store/LoadLogicalNotStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_logical_not_store::GetShapeConfig;
using mlir::afir::load_logical_not_store::kInputShapes;
using mlir::afir::load_logical_not_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadLogicalNotStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  if (input == nullptr || output == nullptr) {
    if (input != nullptr) {
      AscendC::GmFree(input);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  // 三档轮着喂：0、正数、负数。零与非零都要有——全非零的话「结果恒为 0」也能过，
  // 而那正是「常量 block 没被 Duplicate 成 1」的一种表现；负数那一档钉的是包装开头
  // 那个 `Abs`。
  //
  // 非零的取整数且 |x| >= 1：包装算的是 `|1 - min(1, |x|)|`，对 0 与 1 之间的小数它给
  // 的是个小数而不是真值，那是 legacy 自己的语义，不是这条用例要覆的东西。中间要过一趟
  // f32 -> f16，所以取的值都在 half 能精确表示的范围里。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    const float magnitude = static_cast<float>(index % 5U) + 1.0F;
    if (index % 3U == 0U) {
      input[index] = 0.0F;
    } else if (index % 3U == 1U) {
      input[index] = magnitude;
    } else {
      input[index] = -magnitude;
    }
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = -1.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(input);
    AscendC::GmFree(output);
    FAIL() << "ATT tiling failed";
  }
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(input);
      AscendC::GmFree(output);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input),
              reinterpret_cast<uint8_t *>(output), reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    const float expected = input[index] == 0.0F ? 1.0F : 0.0F;
    EXPECT_NEAR(output[index], expected, 1e-3F) << "index=" << index;
  }

  AscendC::GmFree(input);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
