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
#include "test/e2e/cpu/cases/load_axpy_half_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_axpy_half_store::GetShapeConfig;
using mlir::afir::load_axpy_half_store::kAlpha;
using mlir::afir::load_axpy_half_store::kInputShapes;
using mlir::afir::load_axpy_half_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadAxpyHalfStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *lhs = static_cast<half *>(AscendC::GmAlloc(input_element_count * sizeof(half)));
  auto *rhs = static_cast<half *>(AscendC::GmAlloc(input_element_count * sizeof(half)));
  auto *output = static_cast<half *>(AscendC::GmAlloc(output_element_count * sizeof(half)));
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

  // **取值全部在 half 里精确可表示，中间结果和期望值也是**，所以下面可以逐元素比相等
  // 而不是比容差。这一条是有意的：f16 分支把 tmp 切两段做 `Cast` 往返，切错段或者 tmp
  // 不够大时结果会偏一点点——容差一放宽，那种错就正好漏过去。
  //
  // lhs ∈ [-3, 3]（整数），rhs ∈ {2, 4, 6, 8, 10}，alpha = 0.5，于是 alpha*rhs ∈ [1, 5]
  // 是整数，和落在 [-2, 8]。half 的尾数 10 位，这个范围里的整数与半整数都精确。
  //
  // 两个输入喂得不一样：`y = x1 + alpha * x2` 里两个 src 的位置不对称，喂同一份数据
  // 的话把两个实参调换过来照样全绿。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    lhs[index] = half(static_cast<float>(index % 7U) - 3.0F);
    rhs[index] = half((static_cast<float>(index % 5U) + 1.0F) * 2.0F);
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = half(0.0F);
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
    // 走到这里说明 f16 分支里那块 tmp buffer 被切对了、也够大——它是这条路上唯一真正
    // 使用 tmp 的分支，`load_axpy_store`（f32）走的 `else` 一次都不碰它。
    const float expected = static_cast<float>(lhs[index]) + kAlpha * static_cast<float>(rhs[index]);
    EXPECT_FLOAT_EQ(static_cast<float>(output[index]), expected)
        << "index=" << index << " lhs=" << static_cast<float>(lhs[index]) << " rhs=" << static_cast<float>(rhs[index]);
  }

  AscendC::GmFree(lhs);
  AscendC::GmFree(rhs);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
