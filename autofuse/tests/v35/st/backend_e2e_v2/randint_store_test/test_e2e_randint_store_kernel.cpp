/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void randint_store_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR x4, GM_ADDR y,
                                                         GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendRandIntStoreCode : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

// host 侧独立参考实现：与设备侧同一公式（Philox4x32-10 + uint64 取模），不调用被测 AscendC API
static const uint32_t kGoldenPhiloxMul0 = 0xD2511F53u;
static const uint32_t kGoldenPhiloxMul1 = 0xCD9E8D57u;
static const uint32_t kGoldenPhiloxKeyAdd0 = 0x9E3779B9u;
static const uint32_t kGoldenPhiloxKeyAdd1 = 0xBB67AE85u;

static void GoldenPhiloxRound(uint32_t *counter, const uint32_t *key) {
  const uint64_t res0 = static_cast<uint64_t>(kGoldenPhiloxMul0) * static_cast<uint64_t>(counter[0]);
  const uint64_t res1 = static_cast<uint64_t>(kGoldenPhiloxMul1) * static_cast<uint64_t>(counter[2]);
  const uint32_t res0Lo = static_cast<uint32_t>(res0 & 0xFFFFFFFFu);
  const uint32_t res0Hi = static_cast<uint32_t>(res0 >> 32U);
  const uint32_t res1Lo = static_cast<uint32_t>(res1 & 0xFFFFFFFFu);
  const uint32_t res1Hi = static_cast<uint32_t>(res1 >> 32U);
  counter[0] = res1Hi ^ counter[1] ^ key[0];
  counter[1] = res1Lo;
  counter[2] = res0Hi ^ counter[3] ^ key[1];
  counter[3] = res0Lo;
}

static int64_t GoldenRandInt(int64_t seed, int32_t counter, int64_t low, int64_t high) {
  uint32_t key[2] = {static_cast<uint32_t>(static_cast<uint64_t>(seed) & 0xFFFFFFFFu),
                     static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32U)};
  uint32_t counterTmp[4] = {static_cast<uint32_t>(counter), 0, 0, 0};
  for (uint32_t i = 0; i < 10; i++) {
    GoldenPhiloxRound(counterTmp, key);
    key[0] += kGoldenPhiloxKeyAdd0;
    key[1] += kGoldenPhiloxKeyAdd1;
  }
  const uint64_t randU64 = (static_cast<uint64_t>(counterTmp[0]) << 32U) | static_cast<uint64_t>(counterTmp[1]);
  const uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low);
  return low + static_cast<int64_t>(randU64 % range);
}

TEST_P(E2EBackendRandIntStoreCode, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];
  const int64_t low = 0;
  const int64_t high = 24;
  const int64_t seed = 42;

  AutofuseTilingData tiling_data;
  // seed 输入与真实前端 extern 采样的 buf0 同形态：仅 1 个 int64（标量广播，Load 视图 repeats 全 1）
  int64_t *x1 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) + 32));
  int32_t *x2 = static_cast<int32_t *>(AscendC::GmAlloc(test_size * sizeof(int32_t) + 32));
  int64_t *x3 = static_cast<int64_t *>(AscendC::GmAlloc(test_size * sizeof(int64_t) + 32));
  int64_t *x4 = static_cast<int64_t *>(AscendC::GmAlloc(test_size * sizeof(int64_t) + 32));
  int64_t *y = static_cast<int64_t *>(AscendC::GmAlloc(test_size * sizeof(int64_t) + 32));
  int64_t *expect = static_cast<int64_t *>(AscendC::GmAlloc(test_size * sizeof(int64_t) + 32));

  x1[0] = seed;
  for (int i = 0; i < test_size; i++) {
    x2[i] = i;
    x3[i] = low;
    x4[i] = high;
    expect[i] = GoldenRandInt(seed, i, low, high);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(randint_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(x1),
              reinterpret_cast<uint8_t *>(x2), reinterpret_cast<uint8_t *>(x3), reinterpret_cast<uint8_t *>(x4),
              reinterpret_cast<uint8_t *>(y), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if ((y[i] != expect[i]) || (y[i] < low) || (y[i] >= high)) {
      diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0U) << " of " << test_size;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(x3);
  AscendC::GmFree(x4);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendRandIntStoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
