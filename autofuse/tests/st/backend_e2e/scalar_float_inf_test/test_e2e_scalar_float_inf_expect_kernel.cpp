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
extern "C" __global__ __aicore__ void scalar_inf_add_test(GM_ADDR x1, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData *tiling,
                                  uint64_t *workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendScalarFloatInf_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendScalarFloatInf_Code, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];
  AutofuseTilingData tiling_data;
  float *input1 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  // 构造float的inf值
  union ScalarBitcodeFloat {
    __aicore__ ScalarBitcodeFloat() {}
    float input;
    uint32_t output;
  } data_float;
  data_float.output = 0x7F800000U;

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_size; i++) {
    input1[i] = rand() / (double)RAND_MAX;
    expect[i] = static_cast<float>(data_float.input);
  }

  // Launch
  uint64_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], test_shape[2], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(scalar_inf_add_test, tiling_data.block_dim, (uint8_t *)input1, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendScalarFloatInf_Code,
                         ::testing::Values(std::vector<int>{32, 16, 16},  // 用例输入的维度需要与构图接口的dims_size匹配
                                           std::vector<int>{32, 16, 18}, std::vector<int>{32, 512, 15}));
