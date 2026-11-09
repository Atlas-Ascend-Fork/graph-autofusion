/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void transpose_int8_test(GM_ADDR x1, GM_ADDR y1, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData *tiling,
                                  uint64_t *workspaceSize, uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

// 图语义: 输入 [s0][s1][s2] -> 输出 [s1][s0][s2], out[a][b][c] = in[b][a][c]
class E2EBackendTransposeInt8Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2EBackendTransposeInt8Code, CalculateCorrect) {
  // CPU模拟器不支持int8数据搬运指令（nddma/DataCopyPad路径均报pv_ccu unknown instr->pipe()），需上板执行
  GTEST_SKIP() << "CPU simulator does not support int8 data-move instructions (pv_ccu: unknown instr->pipe())";
  auto test_shape = GetParam();
  uint32_t block_dim = 48;
  const int s0 = test_shape[0];
  const int s1 = test_shape[1];
  const int s2 = test_shape[2];
  int test_size = s0 * s1 * s2;

  AutofuseTilingData tiling_data;
  int8_t *x1 = static_cast<int8_t *>(AscendC::GmAlloc(test_size * sizeof(int8_t) + 32));
  int8_t *y1 = static_cast<int8_t *>(AscendC::GmAlloc(test_size * sizeof(int8_t) + 32));
  int8_t *expect = static_cast<int8_t *>(AscendC::GmAlloc(test_size * sizeof(int8_t) + 32));

  for (int i = 0; i < test_size; i++) {
    x1[i] = static_cast<int8_t>((i * 37 + 11) % 256 - 128);
    y1[i] = 0;
  }
  for (int a = 0; a < s1; a++) {
    for (int b = 0; b < s0; b++) {
      for (int c = 0; c < s2; c++) {
        expect[a * s0 * s2 + b * s2 + c] = x1[b * s1 * s2 + a * s2 + c];
      }
    }
  }

  uint64_t ws_size = 0;
  AutofuseTiling(s0, s1, s2, &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(transpose_int8_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x1),
              reinterpret_cast<uint8_t *>(y1), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y1[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x1);
  AscendC::GmFree(y1);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendTransposeInt8Code,
                         ::testing::Values(std::vector<int>{4, 8, 3}, std::vector<int>{33, 18, 5},
                                           std::vector<int>{16, 512, 7}));
