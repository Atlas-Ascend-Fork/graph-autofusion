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
#include "test/e2e/cpu/cases/load_broadcast_bf16_reg_base_store/LoadBroadcastBf16RegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_broadcast_bf16_reg_base_store::GetShapeConfig;
using mlir::afir::load_broadcast_bf16_reg_base_store::kInputShapes;
using mlir::afir::load_broadcast_bf16_reg_base_store::kOutputShapes;

// **宿主侧按 `uint16_t` 摆这块内存，比的是位。**
//
// 广播是纯搬运，不做任何算术，所以逐位相等是比「近似相等」更紧的判据——近似比对会把
// 「少搬了一个元素」之外的那一类错（写进去的是转过一次 f32 的值）放过去。而按位比不需要
// 宿主侧有一个 bf16 的标量类；kernel 那一侧的类型由生成码里的 `bfloat16_t` 说，
// `Case.cpp` 的 `CheckGeneratedKernel` 钉着它。
//
// bf16 的位型就是 f32 的高 16 位，所以取值直接从 f32 截断得到（低 16 位全 0 的 f32 在
// bf16 里精确可表示，截断不引入舍入）。
uint16_t Bf16Bits(float value) {
  uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value), "float must be 4 bytes");
  __builtin_memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16U);
}
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadBroadcastBf16RegBaseStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *input0 = static_cast<uint16_t *>(AscendC::GmAlloc(input_element_count * sizeof(uint16_t)));
  auto *output = static_cast<uint16_t *>(AscendC::GmAlloc(output_element_count * sizeof(uint16_t)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // **一半取负、一个都不取 0。** 全正的话「搬错了符号位」看不出来；取 0 的话输出哨兵与
  // 「真的搬进来一个 0」在那几个下标上分不开。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    const float magnitude = static_cast<float>(index % 13U) + 0.5F;
    input0[index] = Bf16Bits((index % 2U == 0U) ? magnitude : -magnitude);
  }
  // **哨兵，不是 0。** 这一档的 bf16 位是刚开的，「kernel 一个字都没写」正是要防的失效
  // 方式——填 0 的话，一个没写过的下标与一个搬进来的 0 长得一样。
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
    EXPECT_EQ(output[index], input0[inputIndex]) << "index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
