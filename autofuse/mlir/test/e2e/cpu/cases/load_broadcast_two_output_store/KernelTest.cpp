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
#include "test/e2e/cpu/cases/load_broadcast_two_output_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_broadcast_two_output_store::BroadcastSourceIndex;
using mlir::afir::load_broadcast_two_output_store::GetShapeConfig;
using mlir::afir::load_broadcast_two_output_store::kInputShapes;
using mlir::afir::load_broadcast_two_output_store::kOutputShapes;
}  // namespace

// 两个输出各占一个形参，次序与 `Output` 的 index 一致（实测生成的签名就是
// `(input0, output0, output1, workspace, tiling)`）。少写一个的后果不是编译错，是
// `output1` 收到 workspace 的地址——**所以这份用例必须两个输出都比**。
extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output0, GM_ADDR output1,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadBroadcastTwoOutputCalculatesCorrectResult) {
  const uint32_t inputCount = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output0Count = GetShapeConfig(kOutputShapes, 0).element_count;
  const uint32_t output1Count = GetShapeConfig(kOutputShapes, 1).element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(inputCount * sizeof(float)));
  float *output0 = static_cast<float *>(AscendC::GmAlloc(output0Count * sizeof(float)));
  float *output1 = static_cast<float *>(AscendC::GmAlloc(output1Count * sizeof(float)));
  if (input0 == nullptr || output0 == nullptr || output1 == nullptr) {
    if (input0 != nullptr) {
      AscendC::GmFree(input0);
    }
    if (output0 != nullptr) {
      AscendC::GmFree(output0);
    }
    if (output1 != nullptr) {
      AscendC::GmFree(output1);
    }
    FAIL() << "input/output allocation failed";
  }

  for (uint32_t index = 0; index < inputCount; ++index) {
    input0[index] = static_cast<float>(index) * 0.25F - 3.0F;
  }
  // 两个输出的初值**不同**：某一条 Store 没发出来时，留下的初值能指出是哪一条。
  for (uint32_t index = 0; index < output0Count; ++index) {
    output0[index] = -1.0F;
  }
  for (uint32_t index = 0; index < output1Count; ++index) {
    output1[index] = -2.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(input0);
    AscendC::GmFree(output0);
    AscendC::GmFree(output1);
    FAIL() << "ATT tiling failed";
  }
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(input0);
      AscendC::GmFree(output0);
      AscendC::GmFree(output1);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input0),
              reinterpret_cast<uint8_t *>(output0), reinterpret_cast<uint8_t *>(output1),
              reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < output0Count; ++index) {
    const uint32_t source =
        BroadcastSourceIndex(GetShapeConfig(kOutputShapes, 0).shape, GetShapeConfig(kInputShapes, 0).shape, index);
    EXPECT_FLOAT_EQ(output0[index], input0[source]) << "output0 index=" << index;
    EXPECT_FLOAT_EQ(output1[index], input0[source]) << "output1 index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output0);
  AscendC::GmFree(output1);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
