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
extern "C" __global__ __aicore__ void concat_to_stores_test(GM_ADDR data0, GM_ADDR data1, GM_ADDR output,
                                                            GM_ADDR workspace, GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint64_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendConcatToStores_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendConcatToStores_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  int input_size = test_shape[0] * test_shape[1];
  int output_size = input_size * 2;

  AutofuseTilingData tiling_data;
  int32_t *input1 = (int32_t *)AscendC::GmAlloc(input_size * sizeof(int32_t) + 32);
  int32_t *input2 = (int32_t *)AscendC::GmAlloc(input_size * sizeof(int32_t) + 32);
  int32_t *y = (int32_t *)AscendC::GmAlloc(output_size * sizeof(int32_t) + 32);
  int32_t *expect = (int32_t *)AscendC::GmAlloc(output_size * sizeof(int32_t) + 32);

  // Prepare test and expect data
  int32_t value = 0;
  for (int i = 0; i < test_shape[0]; ++i) {
    for (int j = 0; j < test_shape[1]; ++j) {
      input1[i * test_shape[1] + j] = value;
      input2[i * test_shape[1] + j] = 100000 + value;
      expect[i * 2 * test_shape[1] + j] = input1[i * test_shape[1] + j];
      expect[i * 2 * test_shape[1] + test_shape[1] + j] = input2[i * test_shape[1] + j];
      value++;
    }
  }

  // Launch
  uint64_t ws_size = 0;
  AutofuseTiling(&tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.graph0_tiling_key, tiling_data.block_dim);
  EXPECT_EQ(tiling_data.graph0_tiling_key, 1U);
  EXPECT_EQ(tiling_data.graph0_result1_g0_tiling_data.ub_size, 0U);
  EXPECT_NE(tiling_data.graph0_result1_g1_tiling_data.ub_size, 0U);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(concat_to_stores_test, tiling_data.block_dim, (uint8_t *)input1, (uint8_t *)input2, (uint8_t *)y, nullptr,
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

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendConcatToStores_Code,
                         ::testing::Values(std::vector<int>{100, 51}));
