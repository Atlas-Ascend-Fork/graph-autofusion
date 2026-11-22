/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <cmath>
#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void load_erfinv_store_test(GM_ADDR x1, GM_ADDR y1, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendLoadErfinvStoreCode : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

static constexpr size_t ERFINV_INPUT_NUM = 13;
static constexpr float ERFINV_INPUTS[ERFINV_INPUT_NUM] = {
    0.0F, 0.5F, -0.5F, 0.9F, -0.9F, 0.99F, -0.99F, 0.999F, -0.999F, 0.9999F, -0.9999F, 0.9998279F, -0.9998279F};
// 期望值按输入 float32 字面量对应的精确值计算：erfinv(double(float32(x))) 后四舍五入到 float32。
// 近 ±1 处 erfinv 导数很大，不能用十进制字面量的 erfinv 作为期望值。
static constexpr float ERFINV_EXPECTS[ERFINV_INPUT_NUM] = {
    0.0F,         0.476936281F,  -0.476936281F, 1.163087130F,  -1.163087130F, 1.821386576F, -1.821386576F,
    2.326756239F, -2.326756239F, 2.751035452F,  -2.751035452F, 2.656480789F,  -2.656480789F};
// 与单算子 SIMT erfinvf 对齐，覆盖近边界输入；容差按该 log2 近似算法的实际误差设定。
static constexpr float ERFINV_TOL = 1e-4F;

TEST_P(E2EBackendLoadErfinvStoreCode, CalculateCorrect) {
  auto erfinv_shape = GetParam();
  uint64_t erfinv_block_dim = 48;
  int erfinv_test_size = erfinv_shape[0] * erfinv_shape[1];
  AutofuseTilingData tiling_data;
  float *erfinv_x = static_cast<float *>(AscendC::GmAlloc(erfinv_test_size * sizeof(float) + 32));
  float *erfinv_y = static_cast<float *>(AscendC::GmAlloc(erfinv_test_size * sizeof(float) + 32));
  float *erfinv_expect = static_cast<float *>(AscendC::GmAlloc(erfinv_test_size * sizeof(float) + 32));

  for (int i = 0; i < erfinv_test_size; i++) {
    erfinv_x[i] = ERFINV_INPUTS[i % ERFINV_INPUT_NUM];
    erfinv_expect[i] = ERFINV_EXPECTS[i % ERFINV_INPUT_NUM];
  }

  uint32_t ws_size = 0;
  AutofuseTiling(erfinv_shape[0], erfinv_shape[1], &tiling_data, &ws_size, &erfinv_block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_erfinv_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(erfinv_x),
              reinterpret_cast<uint8_t *>(erfinv_y), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t erfinv_diff_count = 0;
  for (int i = 0; i < erfinv_test_size; i++) {
    if (std::fabs(erfinv_y[i] - erfinv_expect[i]) > ERFINV_TOL) {
      erfinv_diff_count++;
    }
  }
  EXPECT_EQ(erfinv_diff_count, 0U) << " of " << erfinv_test_size;
  AscendC::GmFree(erfinv_x);
  AscendC::GmFree(erfinv_y);
  AscendC::GmFree(erfinv_expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendLoadErfinvStoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
