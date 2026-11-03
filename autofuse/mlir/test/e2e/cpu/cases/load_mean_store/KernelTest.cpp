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
#include <vector>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_mean_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_mean_store::GetShapeConfig;
using mlir::afir::load_mean_store::kInputShapes;
using mlir::afir::load_mean_store::kOutputShapes;

// 参考值用 `double` 累加再除，与被测 kernel 的 f32 累加次序无关。两边都对的时候差的只是
// f32 的舍入；而本用例要区分的三类错（漏除、重复除、分母错）都是**量级**上的差别，不是
// 舍入量级的差别，所以这个容差既足够松、又足够区分。
//
// 为什么不用 `EXPECT_FLOAT_EQ`（和那边的求和用例一样）：求和那边输入取的是小整数，每个
// 部分和都在 f32 里精确可表示；均值多了一次除法，`1/37` 这类分母本身就不精确，再乘回去
// 必然与 `double` 参考差几个 ulp。写死 `FLOAT_EQ` 会在 R 不是 2 的幂的那几格假报。
constexpr double kRelativeTolerance = 1e-5;

void ExpectClose(float actual, double expected, const std::string &where) {
  const double tolerance = kRelativeTolerance * std::max(1.0, std::abs(expected));
  EXPECT_NEAR(static_cast<double>(actual), expected, tolerance) << where;
}
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadMeanStoreCalculatesCorrectResult) {
  const auto &input_config = GetShapeConfig(kInputShapes, 0);
  const uint32_t input_element_count = input_config.element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  const std::vector<int64_t> input_shape(input_config.shape);
  const std::vector<int64_t> output_shape(GetShapeConfig(kOutputShapes, 0).shape);
  ASSERT_EQ(input_shape.size(), 2U) << "load_mean_store 只注册二维形状";
  const bool reduces_leading_axis = output_shape[0] == 1;

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

  // 小整数让每个部分和在 f32 里精确可表示，于是误差全部来自那一次除法。
  // 取值跨正负，均值接近 0 而和不接近 0：漏除一次会让结果偏大 R 倍，重复除会让它塌向 0，
  // 两种都远在容差之外。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = static_cast<float>(index % 7U) - 3.0F;
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
#ifdef AF_REDUCE_PARTITION
  ASSERT_GT(workspaceSize, 0U) << "partitioned reduction must use GM workspace";
#endif
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

  const uint32_t rows = static_cast<uint32_t>(input_shape[0]);
  const uint32_t columns = static_cast<uint32_t>(input_shape[1]);
  ASSERT_EQ(rows * columns, input_element_count);

  // 分母恒为**完整**归约域的元素数：AR 下是列数、RA 下是行数。它与 tile 怎么切、切出几轮、
  // 尾块多长都无关——这正是本组用例要钉的东西。
  if (reduces_leading_axis) {
    // RA：`[R, A]`，沿列求均值。
    ASSERT_EQ(output_element_count, columns);
    for (uint32_t column = 0; column < columns; ++column) {
      double total = 0.0;
      for (uint32_t row = 0; row < rows; ++row) {
        total += static_cast<double>(input0[row * columns + column]);
      }
      ExpectClose(output[column], total / static_cast<double>(rows), "column=" + std::to_string(column));
    }
  } else {
    // AR：`[A, R]`，沿行求均值。
    ASSERT_EQ(output_element_count, rows);
    for (uint32_t row = 0; row < rows; ++row) {
      double total = 0.0;
      for (uint32_t column = 0; column < columns; ++column) {
        total += static_cast<double>(input0[row * columns + column]);
      }
      ExpectClose(output[row], total / static_cast<double>(columns), "row=" + std::to_string(row));
    }
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
