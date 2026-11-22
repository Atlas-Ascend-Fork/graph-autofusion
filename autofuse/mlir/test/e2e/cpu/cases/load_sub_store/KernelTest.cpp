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
#include "test/e2e/cpu/cases/load_sub_store/LoadSubStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_sub_store::GetShapeConfig;
using mlir::afir::load_sub_store::kInputShapes;
using mlir::afir::load_sub_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadSubStoreCalculatesCorrectResult) {
  const uint32_t input0_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t input1_element_count = GetShapeConfig(kInputShapes, 1).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(input0_element_count * sizeof(float)));
  float *input1 = static_cast<float *>(AscendC::GmAlloc(input1_element_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  if (input0 == nullptr || input1 == nullptr || output == nullptr) {
    if (input0 != nullptr) {
      AscendC::GmFree(input0);
    }
    if (input1 != nullptr) {
      AscendC::GmFree(input1);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  // 两边都取得有正有负，而且两个输入不相等：相减以后全 0 的话，一份把 `Sub` 写成
  // `Add` 的实现也可能凑巧过——差值恒不为零才分得开。
  for (uint32_t index = 0; index < input0_element_count; ++index) {
    input0[index] = static_cast<float>(static_cast<int32_t>(index % 9U) - 4) * 1.5F;
  }
  // 减数跳过 0：下面那条「不等于 a + b」的判据在 b == 0 的位置恒不成立
  // （`a - 0` 与 `a + 0` 本来就相等），留着 0 就是给自己造一个假红。
  // 取值落在 {-3, -2, -1, 1, 2} * 0.25。
  for (uint32_t index = 0; index < input1_element_count; ++index) {
    int32_t step = static_cast<int32_t>(index % 5U) - 3;
    if (step >= 0) {
      step += 1;
    }
    input1[index] = static_cast<float>(step) * 0.25F;
  }

  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(input0);
    AscendC::GmFree(input1);
    AscendC::GmFree(output);
    FAIL() << "ATT tiling failed";
  }
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(input0);
      AscendC::GmFree(input1);
      AscendC::GmFree(output);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input0),
              reinterpret_cast<uint8_t *>(input1), reinterpret_cast<uint8_t *>(output),
              reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    const float expected = input0[index] - input1[index];
    EXPECT_NEAR(output[index], expected, 1e-5F) << "index=" << index;
    // 写成 `Add` 的话上一条在 input1 为 0 的位置照样过，这一条不会——**前提是减数不为
    // 0**，见上面构造输入那段。
    EXPECT_NE(expected, input0[index] + input1[index]) << "index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
