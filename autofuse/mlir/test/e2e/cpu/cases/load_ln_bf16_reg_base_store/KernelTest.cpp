/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_ln_bf16_reg_base_store/LoadLnBf16RegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_ln_bf16_reg_base_store::GetShapeConfig;
using mlir::afir::load_ln_bf16_reg_base_store::kInputShapes;
using mlir::afir::load_ln_bf16_reg_base_store::kOutputShapes;

// 宿主侧按 `uint16_t` 摆这块内存。bf16 的位型就是 f32 的高 16 位，所以两个方向都不需要
// 宿主侧有 bf16 的标量类。
uint16_t Bf16Bits(float value) {
  uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value), "float must be 4 bytes");
  __builtin_memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16U);
}

float Bf16ToFloat(uint16_t bits) {
  const uint32_t wide = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  __builtin_memcpy(&value, &wide, sizeof(value));
  return value;
}
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadLnBf16RegBaseStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *input0 = static_cast<uint16_t *>(AscendC::GmAlloc(input_element_count * sizeof(uint16_t)));
  auto *output = static_cast<uint16_t *>(AscendC::GmAlloc(output_element_count * sizeof(uint16_t)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // 输入取 1.25 到 3.25 的四分之一整数：都在 bf16 里精确可表示，所以宿主侧的参考值只带
  // 一次 f32 对数的舍入。不取 1——ln(1) = 0，相对误差在那里没有定义。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = Bf16Bits(1.25F + 0.25F * static_cast<float>(index % 9U));
  }
  // 哨兵是个负数：这一组输入的对数恒正，一个字都没写时一眼看得出。
  const uint16_t sentinel = Bf16Bits(-12345.0F);
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = sentinel;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  auto *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0U ? 1U : workspaceSize));
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input0),
              reinterpret_cast<uint8_t *>(output), workspace, tilingData);

  // 判据是相对误差 1%：bf16 的相对精度是 2^-8（约 0.4%），结果从 f32 落回 bf16 时无论
  // 舍入还是截断都落在真值的一个 ulp 之内。这一组真值不小于 0.22，所以相对误差够用。
  // 近似比对逮不住「差一个 ulp」，但逮得住这一格真正的失效方式——换错了算子、lane 宽度
  // 取错、dist 没带上。
  for (uint32_t index = 0; index < output_element_count; ++index) {
    const float expected = std::log(Bf16ToFloat(input0[index]));
    EXPECT_NEAR(Bf16ToFloat(output[index]), expected, expected * 0.01F) << "element " << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
