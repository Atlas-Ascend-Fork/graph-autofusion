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
#include "test/e2e/cpu/cases/load_broadcast_nddma_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_broadcast_nddma_store::GetShapeConfig;
using mlir::afir::load_broadcast_nddma_store::kInputShapes;
using mlir::afir::load_broadcast_nddma_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadBroadcastNddmaCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  if (input0 == nullptr || output == nullptr) {
    if (input0 != nullptr) {
      AscendC::GmFree(input0);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = static_cast<float>(index) * 0.25F - 3.0F;
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
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

  const std::vector<int64_t> in_dims(GetShapeConfig(kInputShapes, 0).shape.begin(),
                                     GetShapeConfig(kInputShapes, 0).shape.end());
  const std::vector<int64_t> out_dims(GetShapeConfig(kOutputShapes, 0).shape.begin(),
                                      GetShapeConfig(kOutputShapes, 0).shape.end());
  for (uint32_t index = 0; index < output_element_count; ++index) {
    uint32_t remaining = index;
    uint32_t inputIndex = 0;
    uint32_t inputStride = 1;
    for (size_t reverse = out_dims.size(); reverse > 0U; --reverse) {
      const size_t axis = reverse - 1U;
      const uint32_t coordinate = remaining % static_cast<uint32_t>(out_dims[axis]);
      remaining /= static_cast<uint32_t>(out_dims[axis]);
      if (in_dims[axis] != 1) {
        inputIndex += coordinate * inputStride;
      }
      inputStride *= static_cast<uint32_t>(in_dims[axis]);
    }
    EXPECT_FLOAT_EQ(output[index], input0[inputIndex]) << "index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
