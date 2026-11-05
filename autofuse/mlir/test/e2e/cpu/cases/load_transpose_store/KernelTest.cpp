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
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <cstdio>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_transpose_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_transpose_store::GetShapeConfig;
using mlir::afir::load_transpose_store::kInputShapes;
using mlir::afir::load_transpose_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);
extern "C" std::string GetTilingDataRepr(const AutofuseTilingData *tiling);

// `ConfusionTranspose` 的 host tiling 不需要测试侧代填：ATT 的 `api_tiling_gen`（注册键
// "Transpose"，`att/gen_model_info/api_tiling_gen/api/confusion_transpose.cpp`）把
// `GetConfusionTransposeTilingInfo` 的定义与调用直接生成为 tiling 源码的一部分，字段名与
// bridge 写进 func 清单的同一规则（`<node>_tilingData_<tiling_case_id>`）。这里只管灌数、
// 跑、比对。
TEST(MlirE2eCpu, LoadTransposeStoreCalculatesCorrectResult) {
  const auto input_config = GetShapeConfig(kInputShapes, 0);
  const auto output_config = GetShapeConfig(kOutputShapes, 0);
  const uint32_t input_element_count = input_config.element_count;
  const uint32_t output_element_count = output_config.element_count;
  float *input0 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  if (input0 == nullptr || output == nullptr) {
    if (input0 != nullptr) {
      AscendC::GmFree(input0);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = static_cast<float>(index) * 0.25F - 3.0F;
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(input0);
    AscendC::GmFree(output);
    FAIL() << "ATT tiling failed";
  }
  EXPECT_TRUE(GetTilingDataRepr(nullptr).empty());
  const auto repr = GetTilingDataRepr(&tilingData);
  EXPECT_NE(repr.find(".transpose_tilingData_0 = {"), std::string::npos);
  EXPECT_NE(repr.find(".param17 = "), std::string::npos);
  EXPECT_LT(repr.find(".transpose_tilingData_0 = {"), repr.find(".q0_size = "));
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

  // perm `[0, 2, 1]`：输出坐标 `(b, w, z)`（形状 `s0, s2, s1`）↔ 输入坐标 `(b, z, w)`
  // （形状 `s0, s1, s2`）。逐元素比——stride 或 dims 抄错任何一位，这一圈当场红。
  const std::vector<int64_t> in_dims(input_config.shape);
  const std::vector<int64_t> out_dims(output_config.shape);
  ASSERT_EQ(in_dims.size(), 3U);
  ASSERT_EQ(out_dims.size(), 3U);
  for (uint32_t index = 0; index < output_element_count; ++index) {
    uint32_t remaining = index;
    uint32_t coordinate[3] = {0, 0, 0};
    for (size_t reverse = out_dims.size(); reverse > 0U; --reverse) {
      const size_t axis = reverse - 1U;
      coordinate[axis] = remaining % static_cast<uint32_t>(out_dims[axis]);
      remaining /= static_cast<uint32_t>(out_dims[axis]);
    }
    const uint32_t inputIndex = coordinate[0] * static_cast<uint32_t>(in_dims[1] * in_dims[2]) +
                                coordinate[2] * static_cast<uint32_t>(in_dims[2]) + coordinate[1];
    EXPECT_FLOAT_EQ(output[index], input0[inputIndex]) << "index=" << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
