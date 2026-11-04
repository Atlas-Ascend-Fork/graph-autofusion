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
#include "test/e2e/cpu/cases/load_add_sum_store/LoadAddSumStoreCase.h"
#include "test/e2e/cpu/support/E2eTilingShape.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_add_sum_store::GetShapeConfig;
using mlir::afir::load_add_sum_store::kInputShapes;
using mlir::afir::load_add_sum_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadAddSumStoreCalculatesCorrectResult) {
  const auto &input_config = GetShapeConfig(kInputShapes, 0);
  const uint32_t input_element_count = input_config.element_count;
  const auto output_config = GetShapeConfig(kOutputShapes, 0);
  const uint32_t output_element_count = output_config.element_count;
  // `ShapeConfig::shape` 是 `initializer_list`，没有下标运算符——先落成 vector 再按维取。
  const std::vector<int64_t> input_shape(input_config.shape);
  ASSERT_EQ(input_shape.size(), 2U);
  // 两份输入必须同形：Add 不走 broadcast，形状不同时被测的就不是这条路径了。
  ASSERT_EQ(GetShapeConfig(kInputShapes, 1).element_count, input_element_count);

  float *input0 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  float *input1 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  ASSERT_TRUE(input0 != nullptr && input1 != nullptr && output != nullptr);

  // 正负混合，并让两份输入的取值互不相同——同值时 `x + y` 退化成 `2x`，接错操作数看不出来。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = static_cast<float>(index % 7U) - 3.0F;
    input1[index] = 0.5F - static_cast<float>(index % 5U);
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  // 传的是 `TILE_ACCESSOR` 所切那根轴的长度（这条注册切 z0，即第 0 维），否则报出来的覆盖是假的。
  const uint32_t tiled_extent = static_cast<uint32_t>(input_shape[0]);
  const mlir::afir::TilingShape tiling_shape = AF_E2E_TILING_SHAPE(tilingData, blockDim, tiled_extent);
  mlir::afir::ReportTilingShape(AF_MLIR_E2E_CPU_CASE_NAME, tiling_shape);
  AF_E2E_ASSERT_TILING_COVERAGE(tiling_shape);
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    ASSERT_NE(workspace, nullptr);
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input0),
              reinterpret_cast<uint8_t *>(input1), reinterpret_cast<uint8_t *>(output),
              reinterpret_cast<uint8_t *>(workspace), tilingData);

  // reference 是 `sum(x + y, dim=1)`：Add 先落在整块 tile 的局部跨度上（两份输入各一份跨度），
  // ReduceSum 再按 AR 归约。
  const uint32_t reduce_extent = static_cast<uint32_t>(input_shape[1]);
  ASSERT_EQ(output_element_count * reduce_extent, input_element_count);
  for (uint32_t row = 0; row < output_element_count; ++row) {
    float expected = 0.0F;
    for (uint32_t column = 0; column < reduce_extent; ++column) {
      const uint32_t index = (row * reduce_extent) + column;
      expected += input0[index] + input1[index];
    }
    EXPECT_FLOAT_EQ(output[row], expected) << "at row " << row;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
