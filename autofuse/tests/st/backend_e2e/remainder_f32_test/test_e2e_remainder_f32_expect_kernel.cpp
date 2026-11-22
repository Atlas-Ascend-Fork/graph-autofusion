/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void remainder_f32_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace,
                                                         GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendRemainderF32_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendRemainderF32_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *input1 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *input2 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_size; i++) {
    float f1 = (float)(rand() % 200) / 10.0f - 10.0f;  // [-10, 10]
    float f2 = (float)(rand() % 190) / 10.0f + 1.0f;   // [1, 20] to avoid division by zero
    input1[i] = f1;
    input2[i] = f2;
    expect[i] = input1[i] - input2[i] * std::floor(input1[i] / input2[i]);
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(remainder_f32_test, tiling_data.block_dim, (uint8_t *)input1, (uint8_t *)input2, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (std::fabs(y[i] - expect[i]) > 1e-5) {
      printf("diff at index %d: x1: %f, x2: %f, y: %f, expect: %f, diff: %f\n", i, input1[i], input2[i], y[i],
             expect[i], std::fabs(y[i] - expect[i]));
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input1);
  AscendC::GmFree(input2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendRemainderF32_Code,
                         ::testing::Values(std::vector<int>{32, 16},  // 用例输入的维度需要与构图接口的dims_size匹配
                                           std::vector<int>{32, 18}, std::vector<int>{32, 256}));
