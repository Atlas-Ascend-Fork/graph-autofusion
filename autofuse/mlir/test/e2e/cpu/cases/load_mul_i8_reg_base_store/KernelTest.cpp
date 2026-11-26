/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_mul_i8_reg_base_store/LoadMulI8RegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_mul_i8_reg_base_store::GetShapeConfig;
using mlir::afir::load_mul_i8_reg_base_store::kInputShapes;
using mlir::afir::load_mul_i8_reg_base_store::kOutputShapes;

// 边界对，下标一一对应。两个乘数取到 int8 的两头（-128、127），乘积分成三类：恰好落在两头
// 的（-128 × 1、127 × 1、-64 × 2），朝两个方向溢出的（-128 × -1 得 128、127 × 127、
// -128 × 127、64 × 2 得 128、-65 × 2 得 -130），以及范围内的普通值。
constexpr int8_t kEdgeLhs[] = {0, 1, -1, -128, 127, -128, 127, -128, -128, 64, -64, -65, 15, 11, 100, -100};
constexpr int8_t kEdgeRhs[] = {0, 127, 127, 1, 1, -1, 127, -128, 127, 2, 2, 2, -8, 12, 1, 1};
constexpr size_t kEdgePairs = sizeof(kEdgeLhs) / sizeof(kEdgeLhs[0]);

// 乘积在 int16 上算，降回 int8 的那一步带饱和：超出 int8 的钳到两头，而不是截掉高位。
int8_t Narrowed(int32_t product) {
  return static_cast<int8_t>(
      std::clamp<int32_t>(product, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()));
}
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadMulI8RegBaseStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *lhs = static_cast<int8_t *>(AscendC::GmAlloc(input_element_count * sizeof(int8_t)));
  auto *rhs = static_cast<int8_t *>(AscendC::GmAlloc(input_element_count * sizeof(int8_t)));
  auto *output = static_cast<int8_t *>(AscendC::GmAlloc(output_element_count * sizeof(int8_t)));
  ASSERT_NE(lhs, nullptr);
  ASSERT_NE(rhs, nullptr);
  ASSERT_NE(output, nullptr);

  // 偶数下标轮上面的边界对，奇数下标取范围内的小乘积——两个乘数都随下标变，结果不恒定。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    if ((index % 2U) == 0U) {
      const size_t slot = (index / 2U) % kEdgePairs;
      lhs[index] = kEdgeLhs[slot];
      rhs[index] = kEdgeRhs[slot];
    } else {
      lhs[index] = static_cast<int8_t>(static_cast<int32_t>(index % 11U) - 5);
      rhs[index] = static_cast<int8_t>(static_cast<int32_t>(index % 7U) - 3);
    }
  }
  // 结果可能是任何一个 int8 值，挑不出一个「不可能是真值」的哨兵；判据是下面逐元素与期望
  // 比对——kernel 一个字都没写时，绝大多数元素都对不上。
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = static_cast<int8_t>(0x5A);
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  auto *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0U ? 1U : workspaceSize));
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(lhs),
              reinterpret_cast<uint8_t *>(rhs), reinterpret_cast<uint8_t *>(output), workspace, tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    const int32_t product = static_cast<int32_t>(lhs[index]) * static_cast<int32_t>(rhs[index]);
    EXPECT_EQ(output[index], Narrowed(product))
        << "element " << index << " lhs=" << static_cast<int32_t>(lhs[index])
        << " rhs=" << static_cast<int32_t>(rhs[index]) << " product=" << product;
  }

  AscendC::GmFree(lhs);
  AscendC::GmFree(rhs);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
