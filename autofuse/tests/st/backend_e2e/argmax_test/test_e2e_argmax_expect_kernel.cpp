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

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void argmax_test(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint64_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendArgMax_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendArgMax_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  int test_size = test_shape[0] * test_shape[1] * test_shape[2];
  int output_size = test_shape[0] * test_shape[1];  // ArgMax reduces the last dimension

  AutofuseTilingData tiling_data;
  float *input = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  int64_t *y = (int64_t *)AscendC::GmAlloc(output_size * sizeof(int64_t) + 32);
  int64_t *expect = (int64_t *)AscendC::GmAlloc(output_size * sizeof(int64_t) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_size; i++) {
    input[i] = rand() / (double)RAND_MAX;
  }

  // Compute expected ArgMax result (reduce on last axis)
  for (int i = 0; i < test_shape[0] * test_shape[1]; i++) {
    int64_t max_idx = 0;
    float max_val = input[i * test_shape[2]];
    for (int j = 1; j < test_shape[2]; j++) {
      if (input[i * test_shape[2] + j] > max_val) {
        max_val = input[i * test_shape[2] + j];
        max_idx = j;
      }
    }
    expect[i] = max_idx;
  }

  // Launch
  uint64_t ws_size = 0;
  AutofuseTiling(&tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(argmax_test, tiling_data.block_dim, (uint8_t *)input, (uint8_t *)y, nullptr, (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < output_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
      if (diff_count <= 10) {
        printf("Mismatch at [%d]: got %ld, expected %ld\n", i, y[i], expect[i]);
      }
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << output_size;

  AscendC::GmFree(input);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendArgMax_Code,
                         ::testing::Values(std::vector<int>{32, 16, 16}));
