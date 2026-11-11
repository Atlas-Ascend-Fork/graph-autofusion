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
#include "test/e2e/cpu/cases/load_broadcast_add_store/LoadBroadcastAddStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_broadcast_add_store::BroadcastSourceIndex;
using mlir::afir::load_broadcast_add_store::GetShapeConfig;
using mlir::afir::load_broadcast_add_store::kInputShapes;
using mlir::afir::load_broadcast_add_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadBroadcastAddCalculatesCorrectResult) {
  const uint32_t sourceCount = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t fullCount = GetShapeConfig(kInputShapes, 1).element_count;
  const uint32_t outputCount = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(sourceCount * sizeof(float)));
  float *input1 = static_cast<float *>(AscendC::GmAlloc(fullCount * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(outputCount * sizeof(float)));
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

  // 将广播源放大 100 倍，与完整形状输入区分，以提高广播下标或实参映射错误的可见性。
  for (uint32_t index = 0; index < sourceCount; ++index) {
    input0[index] = (static_cast<float>(index) + 1.0F) * 100.0F;
  }
  for (uint32_t index = 0; index < fullCount; ++index) {
    input1[index] = static_cast<float>(index) * 0.25F - 3.0F;
  }
  for (uint32_t index = 0; index < outputCount; ++index) {
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

  for (uint32_t index = 0; index < outputCount; ++index) {
    const uint32_t source =
        BroadcastSourceIndex(GetShapeConfig(kOutputShapes, 0).shape, GetShapeConfig(kInputShapes, 0).shape, index);
    EXPECT_FLOAT_EQ(output[index], input1[index] + input0[source]) << "index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
