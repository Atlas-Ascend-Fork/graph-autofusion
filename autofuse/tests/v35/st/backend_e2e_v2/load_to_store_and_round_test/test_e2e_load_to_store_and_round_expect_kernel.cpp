/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
extern "C" __global__ __aicore__ void load_to_store_and_round_test(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                                                   GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData *tiling,
                                  uint64_t *workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendLoadToStoreAndRound_Code : public testing::Test,
                                            public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendLoadToStoreAndRound_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  bfloat16_t *x = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
  bfloat16_t *y1 = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
  bfloat16_t *y2 = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
  bfloat16_t *expect1 = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
  bfloat16_t *expect2 = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);

  // Prepare test and expect data: y1 为 load 直连 store 的原值，y2 为 round 结果
  for (int i = 0; i < test_size; i++) {
    x[i] = static_cast<bfloat16_t>(static_cast<float>(i) / 2.0f - 5.3f);
    expect1[i] = x[i];
    expect2[i] = static_cast<bfloat16_t>(std::rint(static_cast<float>(x[i])));
  }

  // Launch
  uint64_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], test_shape[2], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_to_store_and_round_test, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y1, (uint8_t *)y2, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  const float atol = 1e-2f;
  for (int i = 0; i < test_size; i++) {
    auto diff1 = static_cast<float>(y1[i]) - static_cast<float>(expect1[i]);
    if (diff1 > atol || diff1 < -atol) {
      diff_count++;
      printf("Index %d: y1=%f, expect=%f\n", i, static_cast<float>(y1[i]), static_cast<float>(expect1[i]));
    }
    auto diff2 = static_cast<float>(y2[i]) - static_cast<float>(expect2[i]);
    if (diff2 > atol || diff2 < -atol) {
      diff_count++;
      printf("Index %d: y2=%f, expect=%f\n", i, static_cast<float>(y2[i]), static_cast<float>(expect2[i]));
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y1);
  AscendC::GmFree(y2);
  AscendC::GmFree(expect1);
  AscendC::GmFree(expect2);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendLoadToStoreAndRound_Code,
                         ::testing::Values(std::vector<int>{32, 16, 16},  // 用例输入的维度需要与构图接口的dims_size匹配
                                           std::vector<int>{32, 16, 18}));
