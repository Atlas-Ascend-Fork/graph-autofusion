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
extern "C" __global__ __aicore__ void split_v2_test(GM_ADDR data0, GM_ADDR output1, GM_ADDR output2, GM_ADDR workspace,
                                                    GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendSplit_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendSplit_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  // 图输入为 u8[s0, 2*s1]，沿最后一维 split 成两个 u8[s0, s1] 输出。
  int rows = test_shape[0];
  int cols = test_shape[1];
  int input_size = rows * cols * 2;
  int output_size = rows * cols;

  AutofuseTilingData tiling_data;
  uint8_t *input = (uint8_t *)AscendC::GmAlloc(input_size * sizeof(uint8_t) + 32);
  uint8_t *y1 = (uint8_t *)AscendC::GmAlloc(output_size * sizeof(uint8_t) + 32);
  uint8_t *y2 = (uint8_t *)AscendC::GmAlloc(output_size * sizeof(uint8_t) + 32);
  uint8_t *expect1 = (uint8_t *)AscendC::GmAlloc(output_size * sizeof(uint8_t) + 32);
  uint8_t *expect2 = (uint8_t *)AscendC::GmAlloc(output_size * sizeof(uint8_t) + 32);

  // Prepare test and expect data
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols * 2; ++j) {
      input[i * cols * 2 + j] = static_cast<uint8_t>(i * cols * 2 + j);
    }
  }

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      expect1[i * cols + j] = input[i * cols * 2 + j];
      expect2[i * cols + j] = input[i * cols * 2 + cols + j];
    }
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(rows, cols, &tiling_data, &ws_size, &block_dim, 1, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(split_v2_test, tiling_data.block_dim, (uint8_t *)input, (uint8_t *)y1, (uint8_t *)y2, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < output_size; i++) {
    if (y1[i] != expect1[i]) {
      diff_count++;
    }

    if (y2[i] != expect2[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << output_size * 2;

  AscendC::GmFree(input);
  AscendC::GmFree(y1);
  AscendC::GmFree(y2);
  AscendC::GmFree(expect1);
  AscendC::GmFree(expect2);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendSplit_Code,
                         ::testing::Values(std::vector<int>{16, 100}, std::vector<int>{1, 1}, std::vector<int>{100, 1},
                                           std::vector<int>{2, 2}, std::vector<int>{200, 2}, std::vector<int>{15, 15},
                                           std::vector<int>{16, 16}, std::vector<int>{17, 17}, std::vector<int>{29, 31},
                                           std::vector<int>{30, 32}, std::vector<int>{31, 33},
                                           std::vector<int>{511, 63}, std::vector<int>{1025, 64}));
