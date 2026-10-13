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
extern "C" __global__ __aicore__ void load_unalign_pad_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR workspace,
                                                            GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint64_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendLoadUnalignPad_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendLoadUnalignPad_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  int32_t *input1 = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) * 1000 + 32);
  int32_t *input2 = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) * 1000 + 32);
  int32_t *y1 = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) * 1000 + 32);
  int32_t *expect1 = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) * 1000 + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_size * 2; i++) {
    input1[i] = i % 2 + 1;
    input2[i] = 2;
    expect1[i] = input1[i] + input2[i];
  }

  // Launch
  uint64_t ws_size = 0;
  AutofuseTiling(&tiling_data, &ws_size, &block_dim, 48, 192 * 1024);

  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_unalign_pad_test, tiling_data.block_dim, (uint8_t *)input1, (uint8_t *)input2, (uint8_t *)y1,
              nullptr, (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y1[i] != expect1[i * 2]) {
      diff_count++;
      printf("y1[%d]:%d  expect1[%d]:%d\n", i, y1[i], i, expect1[i * 2]);
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input1);
  AscendC::GmFree(input2);
  AscendC::GmFree(y1);
  AscendC::GmFree(expect1);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendLoadUnalignPad_Code,
                         ::testing::Values(std::vector<int>{10, 5}));
