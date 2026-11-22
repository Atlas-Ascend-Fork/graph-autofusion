/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_transpose_store_2d/LoadTransposeStore2dCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_transpose_store_2d::GetShapeConfig;
using mlir::afir::load_transpose_store_2d::kInputShapes;
using mlir::afir::load_transpose_store_2d::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

// 2D 全对齐 transpose（`(64,64)→(64,64)`，f16）。纯搬运：half 值不经过任何算术，
// 往返 bit 级无损，所以取值随手填、逐元素**精确比**——段切错、block 参数错那种
// 「搬错位置」的错一个都藏不住。
TEST(MlirE2eCpu, LoadTransposeStore2dCalculatesCorrectResult) {
  const auto input_config = GetShapeConfig(kInputShapes, 0);
  const auto output_config = GetShapeConfig(kOutputShapes, 0);
  const uint32_t input_element_count = input_config.element_count;
  const uint32_t output_element_count = output_config.element_count;
  half *input0 = static_cast<half *>(AscendC::GmAlloc(input_element_count * sizeof(half)));
  half *output = static_cast<half *>(AscendC::GmAlloc(output_element_count * sizeof(half)));
  if (input0 == nullptr || output == nullptr) {
    if (input0 != nullptr) {
      AscendC::GmFree(input0);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  // 小整数在 half 里精确可表示；下标模一圈避免大数进尾数不精确的区间。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = half(static_cast<float>(index % 4096U));
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = half(0.0F);
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

  // perm `[1, 0]`：输出坐标 `(o0, o1)`（形状 `s1, s0`）↔ 输入坐标 `(o1, o0)`。
  const std::vector<int64_t> in_dims(input_config.shape);
  const std::vector<int64_t> out_dims(output_config.shape);
  ASSERT_EQ(in_dims.size(), 2U);
  ASSERT_EQ(out_dims.size(), 2U);
  for (uint32_t index = 0; index < output_element_count; ++index) {
    uint32_t remaining = index;
    uint32_t coordinate[2] = {0, 0};
    for (size_t reverse = out_dims.size(); reverse > 0U; --reverse) {
      const size_t axis = reverse - 1U;
      coordinate[axis] = remaining % static_cast<uint32_t>(out_dims[axis]);
      remaining /= static_cast<uint32_t>(out_dims[axis]);
    }
    const uint32_t inputIndex = coordinate[1] * static_cast<uint32_t>(in_dims[1]) + coordinate[0];
    EXPECT_EQ(output[index], input0[inputIndex]) << "index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
