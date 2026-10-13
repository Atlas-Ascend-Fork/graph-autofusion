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
extern "C" __global__ __aicore__ void brc_inline_test(GM_ADDR x, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace,
                                                      GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint64_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendBrcInline_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendBrcInline_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *input0 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *input1 = (float *)AscendC::GmAlloc(test_shape[1] * sizeof(float) + 32);
  float *input2 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_size; i++) {
    if (i < test_shape[1]) {
      input1[i] = i;
    }
    input0[i] = rand() / (double)RAND_MAX;
    input2[i] = rand() / (double)RAND_MAX;

    expect[i] = (input0[i] + input1[i % test_shape[1]]) * input2[i];
  }

  // Launch
  uint64_t ws_size = 0;
  AutofuseTiling(&tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  std::cout << "tiling key is:" << tiling_data.tiling_key << std::endl;
  ICPU_RUN_KF(brc_inline_test, tiling_data.block_dim, (uint8_t *)input0, (uint8_t *)input1, (uint8_t *)input2,
              (uint8_t *)y, nullptr, (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    auto diff = (double)(y[i] - expect[i]);
    if (diff < -1e-5 || diff > 1e-5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(input2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendBrcInline_Code,
                         ::testing::Values(std::vector<int>{320, 32}));
