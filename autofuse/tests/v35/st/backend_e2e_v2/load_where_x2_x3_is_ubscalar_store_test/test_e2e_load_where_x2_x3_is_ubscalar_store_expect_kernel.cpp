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
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void load_where_store_test(GM_ADDR x1, GM_ADDR y, GM_ADDR workspace,
                                                            GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData *tiling,
                                  uint32_t *workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_LoadWhereX2X3IsUbscalarStore_Code
    : public testing::Test,
      public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {};

TEST_P(E2E_LoadWhereX2X3IsUbscalarStore_Code, CalculateCorrect) {
  auto [test_shape, test_tiling] = GetParam();

  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];
  int output_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  uint8_t *x1 = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t));
  // float *x2 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  // float *x3 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(output_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(output_size * sizeof(float) + 32);

  // Prepare test and expect data
  for (int i = 0; i < test_size; i++) {
    x1[i] = 1;
    // x2[i] = 100;
    // x3[i] = 200;
    if (i < output_size) {
      expect[i] = 100;
    }
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], test_shape[2], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_where_store_test, tiling_data.block_dim, (uint8_t *)x1, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < output_size; i++) {
    half diff = y[i] - expect[i];
    printf("y[%d]:%f, expect[%d]:%f\n", i, y[i], i, expect[i]);
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
      diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0) << " of " << output_size;

  AscendC::GmFree(x1);
  // AscendC::GmFree(x2);
  // AscendC::GmFree(x3);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadWhereX2X3IsUbscalarStore_Code,
                         ::testing::Values(std::pair<std::vector<int>, std::vector<int>>{{2, 2, 2}, {3, 3, 3}}));
