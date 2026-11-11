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
#include "test/e2e/cpu/cases/load_add_index_expr_store/LoadAddIndexExprStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_add_index_expr_store::GetShapeConfig;
using mlir::afir::load_add_index_expr_store::kInputShapes;
using mlir::afir::load_add_index_expr_store::kOutputShapes;

// `IndexExpr` 的表达式，和 Case.cpp 里建图时给的那一条是同一个数。
constexpr float kIndexExprValue = 2.0F;
}  // namespace

// 签名里没有 `IndexExpr`。 它不是 kernel 输入——值在 kernel 顶部由 size var 算出来，
// 折成一个常量。这与 `load_add_scalar_data_store` 恰好相反，那边标量占一个形参。
extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadAddIndexExprStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *input0 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // 输入及加上 2.5 后的结果均可被 f32 精确表示，使用逐元素相等比较。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = static_cast<float>(static_cast<int32_t>(index % 9U) - 4) * 0.5F;
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  auto *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0 ? 1 : workspaceSize));
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<GM_ADDR>(input0),
              reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    EXPECT_FLOAT_EQ(output[index], input0[index] + kIndexExprValue) << "element " << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
