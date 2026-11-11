/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_clip_by_value_store/LoadClipByValueStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_clip_by_value_store::GetShapeConfig;
using mlir::afir::load_clip_by_value_store::kInputShapes;
using mlir::afir::load_clip_by_value_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2,
                                                                    GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadClipByValueStoreCalculatesCorrectResult) {
  const uint32_t x_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t lower_count = GetShapeConfig(kInputShapes, 1).element_count;
  const uint32_t upper_count = GetShapeConfig(kInputShapes, 2).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(x_count * sizeof(float)));
  float *input1 = static_cast<float *>(AscendC::GmAlloc(lower_count * sizeof(float)));
  float *input2 = static_cast<float *>(AscendC::GmAlloc(upper_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(input2, nullptr);
  ASSERT_NE(output, nullptr);

  // 输入覆盖低于下界、区间内和高于上界，避免恒等拷贝也能通过。
  // 上下界逐元素变化且不同，用于发现 min / max 操作数映射错误。
  for (uint32_t index = 0; index < x_count; ++index) {
    input0[index] = static_cast<float>(static_cast<int32_t>(index % 9) - 4);
    input1[index] = -1.0F - static_cast<float>(index % 2);
    input2[index] = 1.0F + static_cast<float>(index % 3);
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  uint8_t *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0 ? 1 : workspaceSize));
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<GM_ADDR>(input0),
              reinterpret_cast<GM_ADDR>(input1), reinterpret_cast<GM_ADDR>(input2), reinterpret_cast<GM_ADDR>(output),
              reinterpret_cast<GM_ADDR>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    const float expected = std::min(std::max(input0[index], input1[index]), input2[index]);
    EXPECT_NEAR(output[index], expected, 1e-6F) << "element " << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(input2);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
