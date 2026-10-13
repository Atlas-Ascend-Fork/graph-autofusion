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
#include <cmath>

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void load_pow_all_input_is_scalar_store_test(GM_ADDR scalar1, GM_ADDR scalar2,
                                                                              GM_ADDR output, GM_ADDR workspace,
                                                                              GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

class E2E_LoadPowAllInputIsScalarStore_Code : public testing::Test,
                                              public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_LoadPowAllInputIsScalarStore_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x0 = (float *)AscendC::GmAlloc(1 * sizeof(float) + 32);
  float *x1 = (float *)AscendC::GmAlloc(1 * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  x0[0] = 1.2f;
  x1[0] = 2.2f;

  // Prepare test and expect data
  for (int i = 0; i < test_size; i++) {
    expect[i] = std::pow(x0[0], x1[0]);
  }

  // Launch
  uint64_t ws_size = 0;
  AutofuseTiling(&tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_pow_all_input_is_scalar_store_test, tiling_data.block_dim, (uint8_t *)x0, (uint8_t *)x1,
              (uint8_t *)y, nullptr, (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    float diff = std::fabs(y[i] - expect[i]);
    if (diff > (float)1e-6) {
      diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x0);
  AscendC::GmFree(x1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadPowAllInputIsScalarStore_Code,
                         ::testing::Values(std::vector<int>{2, 8}));
