/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_broadcast_multi_axis_dynamic_store/LoadBroadcastMultiAxisDynamicStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_broadcast_multi_axis_dynamic_store::GetShapeConfig;
using mlir::afir::load_broadcast_multi_axis_dynamic_store::kInputShapes;
using mlir::afir::load_broadcast_multi_axis_dynamic_store::kOutputShapes;

}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadBroadcastMultiAxisDynamicStoreCalculatesCorrectResult) {
  const uint32_t sourceCount = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t outputCount = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *source = static_cast<float *>(AscendC::GmAlloc(sourceCount * sizeof(float)));
  auto *output = static_cast<float *>(AscendC::GmAlloc(outputCount * sizeof(float)));
  ASSERT_NE(source, nullptr);
  ASSERT_NE(output, nullptr);

  for (uint32_t index = 0; index < sourceCount; ++index) {
    source[index] = static_cast<float>(static_cast<int32_t>(index % 17U) - 8) * 0.25F;
  }
  constexpr float kSentinel = -999.0F;
  for (uint32_t index = 0; index < outputCount; ++index) {
    output[index] = kSentinel;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  auto *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0 ? 1 : workspaceSize));
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<GM_ADDR>(source),
              reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(workspace), tilingData);

  for (uint32_t index = 0; index < outputCount; ++index) {
    EXPECT_FLOAT_EQ(output[index], source[0]) << "index=" << index;
  }

  AscendC::GmFree(source);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
