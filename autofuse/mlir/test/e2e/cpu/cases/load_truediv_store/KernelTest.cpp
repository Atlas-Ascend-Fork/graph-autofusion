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
#include "test/e2e/cpu/cases/load_truediv_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_truediv_store::GetShapeConfig;
using mlir::afir::load_truediv_store::kInputShapes;
using mlir::afir::load_truediv_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadTrueDivStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  // 输入与输出**不同 dtype**：i32 进、f32 出。两边按各自的元素宽度分配——这里两者恰好
  // 同宽，所以分配上看不出差别，但类型必须各写各的，不然下面的期望值会按错的位模式读。
  auto *dividend = static_cast<int32_t *>(AscendC::GmAlloc(input_element_count * sizeof(int32_t)));
  auto *divisor = static_cast<int32_t *>(AscendC::GmAlloc(input_element_count * sizeof(int32_t)));
  auto *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  if (dividend == nullptr || divisor == nullptr || output == nullptr) {
    if (dividend != nullptr) {
      AscendC::GmFree(dividend);
    }
    if (divisor != nullptr) {
      AscendC::GmFree(divisor);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  // **商必须带小数，否则这份用例证明不了「真除」。** 整数除法与真除在整除的那些点上
  // 重合，全喂整除的数据的话，把 `TrueDivExtend` 换成一条截断的整数除法照样全绿。
  //
  // 除数取 ±2 / ±4 / ±8：2 的幂，于是 1/除数 在 f32 里精确可表示，商也就精确
  // （`Cast` 到 f32 那一步对这个量级的整数本来就无损）。被除数取 `除数 * k + 1`，
  // 商就是 `k + 1/除数`——**永远不是整数、也永远不是 0**（|1/除数| ≤ 0.5）。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    const int32_t magnitude = static_cast<int32_t>(1U << ((index % 3U) + 1U));
    divisor[index] = (index % 2U == 0U) ? magnitude : -magnitude;
    const int32_t quotient = static_cast<int32_t>(index % 5U) - 2;
    dividend[index] = divisor[index] * quotient + 1;
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(dividend);
    AscendC::GmFree(divisor);
    AscendC::GmFree(output);
    FAIL() << "ATT tiling failed";
  }
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(dividend);
      AscendC::GmFree(divisor);
      AscendC::GmFree(output);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(dividend),
              reinterpret_cast<uint8_t *>(divisor), reinterpret_cast<uint8_t *>(output),
              reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    const float expected = static_cast<float>(dividend[index]) / static_cast<float>(divisor[index]);
    EXPECT_NEAR(output[index], expected, 1e-5F)
        << "index=" << index << " dividend=" << dividend[index] << " divisor=" << divisor[index];
    // **这一条才是 `truediv` 与 `div` 分得开的地方。** C++ 的 `/` 在两个 int32 上是
    // 截断除法，而包装先 `Cast` 成 f32 再除。上面那组数据保证两者永远不等。
    EXPECT_NE(output[index], static_cast<float>(dividend[index] / divisor[index])) << "index=" << index;
  }

  AscendC::GmFree(dividend);
  AscendC::GmFree(divisor);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
