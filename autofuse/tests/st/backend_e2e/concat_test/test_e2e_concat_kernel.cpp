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
extern "C" __global__ __aicore__ void concat_test(GM_ADDR data0, GM_ADDR data1, GM_ADDR output, GM_ADDR workspace,
                                                  GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint64_t *workspace_size,
                                  uint64_t *block_dim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendConcat_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendConcat_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  int32_t rows = test_shape[0];
  int32_t cols = test_shape[1];
  int32_t src_cols = cols;
  int input_size = rows * src_cols;
  int output_size = input_size * 2;

  AutofuseTilingData tiling_data;
  int32_t *input1 = (int32_t *)AscendC::GmAlloc(input_size * sizeof(int32_t) + 32);
  int32_t *input2 = (int32_t *)AscendC::GmAlloc(input_size * sizeof(int32_t) + 32);
  int32_t *y = (int32_t *)AscendC::GmAlloc(output_size * sizeof(int32_t) + 32);
  int32_t *expect = (int32_t *)AscendC::GmAlloc(output_size * sizeof(int32_t) + 32);

  // Prepare test and expect data
  int32_t value = 0;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      input1[i * src_cols + j] = value;
      input2[i * src_cols + j] = 100000 + value;
      expect[i * 2 * cols + j] = input1[i * src_cols + j];
      expect[i * 2 * cols + cols + j] = input2[i * src_cols + j];
      value++;
    }
  }

  // Launch
  uint64_t ws_size = 0;
  uint64_t block_dim = 48;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.graph0_tiling_key, tiling_data.block_dim);
  printf("sub tiling key: %d\n", tiling_data.graph0_result2_g0_tiling_data.tiling_key);
  if (cols % 8 == 0) {
    EXPECT_EQ(tiling_data.graph0_tiling_key, 0);
  } else if (cols < 48) {
    EXPECT_EQ(tiling_data.graph0_tiling_key, 2);
  } else {
    EXPECT_EQ(tiling_data.graph0_tiling_key, 1);
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(concat_test, tiling_data.block_dim, (uint8_t *)input1, (uint8_t *)input2, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < output_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << output_size;

  AscendC::GmFree(input1);
  AscendC::GmFree(input2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendConcat_Code,
                         ::testing::Values(std::vector<int>{100, 1}, std::vector<int>{100, 15},
                                           std::vector<int>{100, 16}, std::vector<int>{100, 17},
                                           std::vector<int>{100, 31}, std::vector<int>{100, 32},
                                           std::vector<int>{100, 33}, std::vector<int>{100, 47},
                                           std::vector<int>{100, 48}, std::vector<int>{100, 49},
                                           std::vector<int>{100, 65}));
