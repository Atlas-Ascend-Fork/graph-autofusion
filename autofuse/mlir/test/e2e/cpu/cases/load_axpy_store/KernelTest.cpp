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
#include "test/e2e/cpu/cases/load_axpy_store/LoadAxpyStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_axpy_store::GetShapeConfig;
using mlir::afir::load_axpy_store::kAlpha;
using mlir::afir::load_axpy_store::kInputShapes;
using mlir::afir::load_axpy_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadAxpyStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *lhs = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *rhs = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  if (lhs == nullptr || rhs == nullptr || output == nullptr) {
    if (lhs != nullptr) {
      AscendC::GmFree(lhs);
    }
    if (rhs != nullptr) {
      AscendC::GmFree(rhs);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  // 两个输入喂得不一样，而且 `x1 != x2`：`y = x1 + alpha * x2` 里两个 src 的位置不
  // 对称，喂同一份数据的话把两个实参调换过来照样全绿。取值都是 2 的幂的小倍数，f32 里
  // 精确可表示，期望值不引入舍入。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    lhs[index] = static_cast<float>(index % 7U) - 3.0F;
    rhs[index] = (static_cast<float>(index % 5U) + 1.0F) * 2.0F;
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(lhs);
    AscendC::GmFree(rhs);
    AscendC::GmFree(output);
    FAIL() << "ATT tiling failed";
  }
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(lhs);
      AscendC::GmFree(rhs);
      AscendC::GmFree(output);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(lhs),
              reinterpret_cast<uint8_t *>(rhs), reinterpret_cast<uint8_t *>(output),
              reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    // alpha 丢了的话 bridge 先报错；alpha 被建成别的值、或者两个 src 调了位置，
    // 这一条才是唯一拦得住的地方。
    const float expected = lhs[index] + kAlpha * rhs[index];
    EXPECT_FLOAT_EQ(output[index], expected) << "index=" << index << " lhs=" << lhs[index] << " rhs=" << rhs[index];
  }

  AscendC::GmFree(lhs);
  AscendC::GmFree(rhs);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
