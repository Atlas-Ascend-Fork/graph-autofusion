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
#include "test/e2e/cpu/cases/load_logical_and_scalar_reg_base_store/LoadLogicalAndScalarRegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_logical_and_scalar_reg_base_store::GetShapeConfig;
using mlir::afir::load_logical_and_scalar_reg_base_store::kInputShapes;
using mlir::afir::load_logical_and_scalar_reg_base_store::kOutputShapes;

// 与建图时给的 Scalar 节点一致。**只能取 0 或 1**，理由见下面喂值那一段。
constexpr float kMask = 1.0F;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadLogicalAndScalarRegBaseStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  // 输入 f32、输出一个字节的真值，两边按各自的元素宽度分配。
  auto *input0 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  auto *output = static_cast<uint8_t *>(AscendC::GmAlloc(output_element_count * sizeof(uint8_t)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // **只喂 0 / 1**，与 `load_logical_and_store` 同一个契约：包装把两个输入 `Cast` 成
  // half 后直接取乘积再 `Cast` 回 `uint8`，它假定输入已经是真值。喂 4 进去出来的是 4。
  //
  // 标量那一侧同样只能取 0 / 1，这里取 1：取 0 的话结果恒为 0，「标量被当成 0 了」与
  // 「整块没算」分不开。取 1 时结果等于 tile 自己——**所以这条用例钉的是「标量确实以 1
  // 参与了乘积、而且 reg 侧这个重载真的编得过并跑得动」，不是「与和或分得开」**：
  // 标量只有两个取值，表达不出那个区分。`load_logical_and_store` 管 and / or 那一条。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = ((index % 3U) == 0U) ? 1.0F : 0.0F;
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0U;
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
    // 结果只有 0 / 1，所以用等号不用容差。
    const auto expected = static_cast<uint8_t>((input0[index] != 0.0F && kMask != 0.0F) ? 1U : 0U);
    EXPECT_EQ(output[index], expected) << "element " << index << " x=" << input0[index];
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
