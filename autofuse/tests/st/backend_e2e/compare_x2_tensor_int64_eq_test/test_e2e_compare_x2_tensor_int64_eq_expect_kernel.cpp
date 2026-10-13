/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"
#include <random>
#include <chrono>

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void compare_test(GM_ADDR data0, GM_ADDR data1, GM_ADDR data2, GM_ADDR output,
                                                   GM_ADDR workspace, GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint64_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendCompareX2TensorInt64Eq_Code : public testing::Test,
                                               public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendCompareX2TensorInt64Eq_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 1;

  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  int64_t *x1 = (int64_t *)AscendC::GmAlloc(test_size * sizeof(int64_t));
  uint8_t *x2 = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t));
  int64_t *x3 = (int64_t *)AscendC::GmAlloc(test_size * sizeof(int64_t));
  uint8_t *y = (uint8_t *)AscendC::GmAlloc(2 * test_size * sizeof(uint8_t));
  uint8_t *expect = (uint8_t *)AscendC::GmAlloc(2 * test_size * sizeof(uint8_t));

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(1, 100);

  // Prepare test and expect data
  for (int i = 0; i < test_size; i++) {
    x1[i] = distribution(generator);
    x2[i] = 100;
    x3[i] = distribution(generator);
  }
  x1[test_shape[0] + 4] = 111;
  x3[test_shape[0] + 4] = 111;
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < 2 * test_shape[1]; j++) {
      if (j < test_shape[1]) {
        if (x1[i * test_shape[1] + j] == x3[i * test_shape[1] + j]) {
          expect[i * 2 * test_shape[1] + j] = 1;
        } else {
          expect[i * 2 * test_shape[1] + j] = 0;
        }
      } else {
        expect[i * 2 * test_shape[1] + j] = 100;
      }
    }
  }

  // Launch
  uint64_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 1, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(compare_test, tiling_data.block_dim, (uint8_t *)x1, (uint8_t *)x2, (uint8_t *)x3, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size * 2; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size * 2;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(x3);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendCompareX2TensorInt64Eq_Code,
                         ::testing::Values(std::vector<int>{32, 7},  // 用例输入的维度需要与构图接口的dims_size匹配
                                           std::vector<int>{32, 150}, std::vector<int>{8, 272}));
