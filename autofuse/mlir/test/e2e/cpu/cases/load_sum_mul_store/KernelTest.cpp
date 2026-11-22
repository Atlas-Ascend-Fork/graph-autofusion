/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_sum_mul_store/LoadSumMulStoreCase.h"
#include "test/e2e/cpu/support/E2eTilingShape.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_sum_mul_store::GetShapeConfig;
using mlir::afir::load_sum_mul_store::kInputShapes;
using mlir::afir::load_sum_mul_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadSumMulStoreCalculatesCorrectResult) {
  const auto &input_config = GetShapeConfig(kInputShapes, 0);
  const uint32_t input_element_count = input_config.element_count;
  const auto output_config = GetShapeConfig(kOutputShapes, 0);
  const uint32_t output_element_count = output_config.element_count;
  const std::vector<int64_t> input_shape(input_config.shape);
  const std::vector<int64_t> output_shape(output_config.shape);
  ASSERT_EQ(input_shape.size(), 2U);
  ASSERT_EQ(output_shape.size(), 2U);

  float *input0 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  ASSERT_TRUE(input0 != nullptr && output != nullptr);

  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = static_cast<float>(index % 7U) - 3.0F;
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  // 本注册未声明 tiling 访问器，AF_E2E_TILING_SHAPE 的 tile / block_inner 为 0，
  // 不能据此推断切分；元素总数仍有效，保留输出供失败诊断。
  const uint32_t tiled_extent = static_cast<uint32_t>(input_shape[0]);
  const mlir::afir::TilingShape tiling_shape = AF_E2E_TILING_SHAPE(tilingData, blockDim, tiled_extent);
  mlir::afir::ReportTilingShape(AF_MLIR_E2E_CPU_CASE_NAME, tiling_shape);
  AF_E2E_ASSERT_TILING_COVERAGE(tiling_shape);
  // 归约结果通过 GM workspace 传给下一子图，因此测试必须分配 workspace。
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    ASSERT_NE(workspace, nullptr);
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input0),
              reinterpret_cast<uint8_t *>(output), reinterpret_cast<uint8_t *>(workspace), tilingData);

  // reference 是 `sum(x, dim=1) * 2`。
  const uint32_t reduce_extent = static_cast<uint32_t>(input_shape[1]);
  ASSERT_EQ(output_element_count * reduce_extent, input_element_count);
  for (uint32_t row = 0; row < output_element_count; ++row) {
    float expected = 0.0F;
    for (uint32_t column = 0; column < reduce_extent; ++column) {
      expected += input0[row * reduce_extent + column];
    }
    EXPECT_FLOAT_EQ(output[row], expected * 2.0F) << "row=" << row;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
