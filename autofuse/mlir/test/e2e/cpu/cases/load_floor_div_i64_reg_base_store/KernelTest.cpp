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
#include <memory>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_floor_div_i64_reg_base_store/LoadFloorDivI64RegBaseStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_floor_div_i64_reg_base_store::GetShapeConfig;
using mlir::afir::load_floor_div_i64_reg_base_store::kInputShapes;
using mlir::afir::load_floor_div_i64_reg_base_store::kOutputShapes;

struct DivisorPair {
  int64_t lhs;
  int64_t rhs;
};

constexpr int64_t kInt64Min = INT64_MIN;
constexpr int64_t kInt64Max = INT64_MAX;
// 没被写到时输出的值，与任何合法的商都不同。
constexpr int64_t kSentinel = 0x5A5A5A5A5A5A5A5ALL;

// 取值铺满向下取整与截断出现差别的那一格（异号且有余数）、两端的极值，以及除 0 输出 0。
// `INT64_MIN / -1` 不在其中：包装对它没有溢出保护，结果未定义，与 legacy 相同。
constexpr DivisorPair kPairs[] = {
    {7, 2},
    {-7, -2},
    {-7, 2},
    {7, -2},
    {-8, 2},
    {8, -2},
    {0, 5},
    {0, -5},
    {5, 1},
    {-5, 1},
    {5, -1},
    {-5, -1},
    {kInt64Max, 1},
    {kInt64Max, 2},
    {kInt64Max, 3},
    {kInt64Max, -1},
    {kInt64Max, -3},
    {kInt64Min, 1},
    {kInt64Min, 2},
    {kInt64Min, 3},
    {kInt64Min, -2},
    {kInt64Min, -3},
    {kInt64Min + 1, -1},
    {kInt64Min + 1, -2},
    {kInt64Max - 1, kInt64Min},
    {kInt64Min, kInt64Max},
    {-1, kInt64Min},
    {1, kInt64Min},
    {1234567890123, 0},
    {-1234567890123, 0},
    {0, 0},
    {1000000007, 97},
    {-1000000007, 97},
    {1000000007, -97},
    {-1000000007, -97},
};
constexpr size_t kPairCount = sizeof(kPairs) / sizeof(kPairs[0]);

// 参考值用 __int128 算出商再向下取整，不借用包装自己的算法；除 0 一格按包装的约定给出。
int64_t ReferenceFloorDiv(int64_t lhs, int64_t rhs) {
  if (rhs == 0) {
    return 0;
  }
  const __int128 dividend = lhs;
  const __int128 divisor = rhs;
  __int128 quotient = dividend / divisor;
  if (dividend % divisor != 0 && ((dividend < 0) != (divisor < 0))) {
    --quotient;
  }
  return static_cast<int64_t>(quotient);
}
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadFloorDivI64RegBaseStoreCalculatesCorrectResult) {
  const auto release = [](void *buffer) { AscendC::GmFree(buffer); };
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  std::unique_ptr<int64_t[], decltype(release)> input0(
      static_cast<int64_t *>(AscendC::GmAlloc(input_element_count * sizeof(int64_t))), release);
  std::unique_ptr<int64_t[], decltype(release)> input1(
      static_cast<int64_t *>(AscendC::GmAlloc(input_element_count * sizeof(int64_t))), release);
  std::unique_ptr<int64_t[], decltype(release)> output(
      static_cast<int64_t *>(AscendC::GmAlloc(output_element_count * sizeof(int64_t))), release);
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(output, nullptr);

  for (uint32_t index = 0; index < input_element_count; ++index) {
    const DivisorPair &pair = kPairs[index % kPairCount];
    input0[index] = pair.lhs;
    input1[index] = pair.rhs;
  }
  std::fill_n(output.get(), output_element_count, kSentinel);

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  std::unique_ptr<uint8_t[], decltype(release)> workspace(
      static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0 ? 1 : workspaceSize)), release);
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<GM_ADDR>(input0.get()),
              reinterpret_cast<GM_ADDR>(input1.get()), reinterpret_cast<GM_ADDR>(output.get()),
              reinterpret_cast<GM_ADDR>(workspace.get()), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    EXPECT_EQ(output[index], ReferenceFloorDiv(input0[index], input1[index])) << "index=" << index;
  }
}
