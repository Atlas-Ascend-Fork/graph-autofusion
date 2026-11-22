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
extern "C" __global__ __aicore__ void cast_uint8_to_int64_test(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                               GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint64_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendCastUint8ToInt64_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

// CastWithTwoTransfer 复用同一块 UB（inter_buf_2 由 inter_buf_1 reinterpret 而来），且本次是变宽方向
// (1B->2B->4B->8B)，写指针领先读指针。最坏情形为「多轮 calc_loop 的最后一轮 + 尾块」。
// 生成内核 tmp_buf 为 8KB，CastWithTwoTransfer 的 buf_max_calc_size = 8192 / sizeof(half) / 2 = 2048 元素：
//   {32, 7}   -> z0z1t_size = 224  : calc_loop = 0，仅尾块路径；
//   {32, 150} -> z0z1t_size = 4800 : calc_loop = 2 且 tail = 704，多轮 + 尾块；
//   {8, 272}  -> z0z1t_size = 2176 : calc_loop = 1 且 tail = 128，多轮 + 尾块。
TEST_P(E2E_BackendCastUint8ToInt64_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 1;

  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  uint8_t *x = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t));
  int64_t *y = (int64_t *)AscendC::GmAlloc(test_size * sizeof(int64_t));
  int64_t *expect = (int64_t *)AscendC::GmAlloc(test_size * sizeof(int64_t));

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(0, 255);

  // Prepare test and expect data
  for (int i = 0; i < test_size; i++) {
    x[i] = static_cast<uint8_t>(distribution(generator));
    expect[i] = static_cast<int64_t>(x[i]);
  }

  // Launch
  uint64_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 1, 192 * 1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(cast_uint8_to_int64_test, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendCastUint8ToInt64_Code,
                         ::testing::Values(std::vector<int>{32, 7},                                // 尾块场景
                                           std::vector<int>{32, 150}, std::vector<int>{8, 272}));  // 多块循环场景
