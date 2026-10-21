/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void bitwise_bool_store_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2,
                                                              GM_ADDR y3, GM_ADDR y4, GM_ADDR workspace,
                                                              GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

namespace {
// 四分支图：y1 = x1 & x2；y2 = x1 | x2；y3 = x1 ^ x2；y4 = ~x1（bool 语义为逻辑非 0<->1）
// bool 与 uint8 同为 1 字节，位模式一致，按 uint8 准备和校验数据
struct BitwiseBoolTestBuffers {
  uint8_t *x1;
  uint8_t *x2;
  uint8_t *y[4];
  uint8_t *expect[4];
};

BitwiseBoolTestBuffers AllocBitwiseBoolBuffers(int test_size) {
  const size_t buf_size = test_size * sizeof(uint8_t) + 32;
  BitwiseBoolTestBuffers bufs;
  bufs.x1 = static_cast<uint8_t *>(AscendC::GmAlloc(buf_size));
  bufs.x2 = static_cast<uint8_t *>(AscendC::GmAlloc(buf_size));
  for (size_t i = 0; i < 4; i++) {
    bufs.y[i] = static_cast<uint8_t *>(AscendC::GmAlloc(buf_size));
    bufs.expect[i] = static_cast<uint8_t *>(AscendC::GmAlloc(buf_size));
  }
  return bufs;
}

// 覆盖 0/1 全部输入组合
void InitBitwiseBoolData(const BitwiseBoolTestBuffers &bufs, int test_size) {
  srand(1);
  for (int i = 0; i < test_size; i++) {
    bufs.x1[i] = static_cast<uint8_t>(rand() % 2);
    bufs.x2[i] = static_cast<uint8_t>(rand() % 2);
    bufs.expect[0][i] = bufs.x1[i] & bufs.x2[i];
    bufs.expect[1][i] = bufs.x1[i] | bufs.x2[i];
    bufs.expect[2][i] = bufs.x1[i] ^ bufs.x2[i];
    bufs.expect[3][i] = bufs.x1[i] ^ 1;
  }
}

void FreeBitwiseBoolBuffers(const BitwiseBoolTestBuffers &bufs) {
  AscendC::GmFree(bufs.x1);
  AscendC::GmFree(bufs.x2);
  for (size_t i = 0; i < 4; i++) {
    AscendC::GmFree(bufs.y[i]);
    AscendC::GmFree(bufs.expect[i]);
  }
}

uint32_t CountBitwiseBoolDiff(const BitwiseBoolTestBuffers &bufs, int test_size) {
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    for (size_t j = 0; j < 4; j++) {
      if (bufs.y[j][i] != bufs.expect[j][i]) {
        diff_count++;
      }
    }
  }
  return diff_count;
}
}  // namespace

class E2E_BackendLoadBitwiseBoolStore_Code : public testing::Test,
                                             public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendLoadBitwiseBoolStore_Code, CalculateCorrect) {
  auto test_shape = GetParam();
  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  auto bufs = AllocBitwiseBoolBuffers(test_size);
  InitBitwiseBoolData(bufs, test_size);

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(bitwise_bool_store_test, tiling_data.block_dim, bufs.x1, bufs.x2, bufs.y[0], bufs.y[1], bufs.y[2],
              bufs.y[3], nullptr, (uint8_t *)&tiling_data);

  EXPECT_EQ(CountBitwiseBoolDiff(bufs, test_size), 0) << " of " << test_size;

  FreeBitwiseBoolBuffers(bufs);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendLoadBitwiseBoolStore_Code,
                         ::testing::Values(std::vector<int>{3, 77}  // 用例输入的维度需要与构图接口的 dims_size 匹配
                                           ));
