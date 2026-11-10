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
#include "test/e2e/cpu/cases/load_broadcast_chain_store/LoadBroadcastChainStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_broadcast_chain_store::BroadcastSourceIndex;
using mlir::afir::load_broadcast_chain_store::GetShapeConfig;
using mlir::afir::load_broadcast_chain_store::kInputShapes;
using mlir::afir::load_broadcast_chain_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadBroadcastChainCalculatesCorrectResult) {
  const uint32_t inputCount = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t outputCount = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(inputCount * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(outputCount * sizeof(float)));
  if (input0 == nullptr || output == nullptr) {
    if (input0 != nullptr) {
      AscendC::GmFree(input0);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  for (uint32_t index = 0; index < inputCount; ++index) {
    input0[index] = static_cast<float>(index) * 0.25F - 3.0F;
  }
  for (uint32_t index = 0; index < outputCount; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(input0);
    AscendC::GmFree(output);
    FAIL() << "ATT tiling failed";
  }
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(input0);
      AscendC::GmFree(output);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input0),
              reinterpret_cast<uint8_t *>(output), reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < outputCount; ++index) {
    const uint32_t source =
        BroadcastSourceIndex(GetShapeConfig(kOutputShapes, 0).shape, GetShapeConfig(kInputShapes, 0).shape, index);
    // 期望值按一次广播算：两趟拆开之后必须和它逐元素相等。
    EXPECT_FLOAT_EQ(output[index], input0[source]) << "index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
