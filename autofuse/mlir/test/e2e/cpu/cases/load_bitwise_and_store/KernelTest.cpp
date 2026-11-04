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
#include "test/e2e/cpu/cases/load_bitwise_and_store/LoadBitwiseAndStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_bitwise_and_store::GetShapeConfig;
using mlir::afir::load_bitwise_and_store::kInputShapes;
using mlir::afir::load_bitwise_and_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadBitwiseAndStoreCalculatesCorrectResult) {
  const uint32_t input0_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t input1_element_count = GetShapeConfig(kInputShapes, 1).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  int16_t *input0 = static_cast<int16_t *>(AscendC::GmAlloc(input0_element_count * sizeof(int16_t)));
  int16_t *input1 = static_cast<int16_t *>(AscendC::GmAlloc(input1_element_count * sizeof(int16_t)));
  int16_t *output = static_cast<int16_t *>(AscendC::GmAlloc(output_element_count * sizeof(int16_t)));
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

  // 两个位模式，挑得让每一位在用例里都被两种取值走到：0x5555 / 0x3333 交替，再逐元素
  // 移一位，于是「同为 1」「一边 1」「同为 0」三种组合都被覆盖。周期只有 16 个元素，
  // 与形状无关——形状再改也不必回来动这里。
  for (uint32_t index = 0; index < input0_element_count; ++index) {
    input0[index] = static_cast<int16_t>(0x5555 ^ static_cast<int16_t>(index));
  }
  for (uint32_t index = 0; index < input1_element_count; ++index) {
    input1[index] = static_cast<int16_t>(0x3333 ^ static_cast<int16_t>(index << 1U));
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0;
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
    EXPECT_EQ(output[index], static_cast<int16_t>(input0[index] & input1[index])) << "index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
