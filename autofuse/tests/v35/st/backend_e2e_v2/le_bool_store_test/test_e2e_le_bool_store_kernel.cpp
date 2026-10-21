/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void le_bool_store_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y1, GM_ADDR y2,
                                                         GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

namespace {
// 双分支图：y1 = le(x1, x2)（双 tensor）；y2 = le(x3, 0)（const scalar）
// bool 与 uint8 同为 1 字节，位模式一致，按 uint8 准备和校验数据
struct LeBoolTestBuffers {
  uint8_t *x1;
  uint8_t *x2;
  uint8_t *x3;
  uint8_t *y1;
  uint8_t *y2;
  uint8_t *expect1;
  uint8_t *expect2;
};

LeBoolTestBuffers AllocLeBoolBuffers(int test_size) {
  const size_t buf_size = test_size * sizeof(uint8_t) + 32;
  return {static_cast<uint8_t *>(AscendC::GmAlloc(buf_size)), static_cast<uint8_t *>(AscendC::GmAlloc(buf_size)),
          static_cast<uint8_t *>(AscendC::GmAlloc(buf_size)), static_cast<uint8_t *>(AscendC::GmAlloc(buf_size)),
          static_cast<uint8_t *>(AscendC::GmAlloc(buf_size)), static_cast<uint8_t *>(AscendC::GmAlloc(buf_size)),
          static_cast<uint8_t *>(AscendC::GmAlloc(buf_size))};
}

// 覆盖 0/1 全部组合
void InitLeBoolData(const LeBoolTestBuffers &bufs, int test_size) {
  srand(1);
  for (int i = 0; i < test_size; i++) {
    bufs.x1[i] = static_cast<uint8_t>(rand() % 2);
    bufs.x2[i] = static_cast<uint8_t>(rand() % 2);
    bufs.x3[i] = static_cast<uint8_t>(rand() % 2);
    bufs.expect1[i] = (bufs.x1[i] <= bufs.x2[i]) ? 1 : 0;
    bufs.expect2[i] = (bufs.x3[i] <= 0) ? 1 : 0;
  }
}

void FreeLeBoolBuffers(const LeBoolTestBuffers &bufs) {
  AscendC::GmFree(bufs.x1);
  AscendC::GmFree(bufs.x2);
  AscendC::GmFree(bufs.x3);
  AscendC::GmFree(bufs.y1);
  AscendC::GmFree(bufs.y2);
  AscendC::GmFree(bufs.expect1);
  AscendC::GmFree(bufs.expect2);
}
}  // namespace

class E2E_BackendLoadLeBoolStore_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendLoadLeBoolStore_Code, CalculateCorrect) {
  auto test_shape = GetParam();
  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  auto bufs = AllocLeBoolBuffers(test_size);
  InitLeBoolData(bufs, test_size);

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(le_bool_store_test, tiling_data.block_dim, bufs.x1, bufs.x2, bufs.x3, bufs.y1, bufs.y2, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (bufs.y1[i] != bufs.expect1[i] || bufs.y2[i] != bufs.expect2[i]) {
      diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  FreeLeBoolBuffers(bufs);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendLoadLeBoolStore_Code,
                         ::testing::Values(std::vector<int>{3, 77}  // 用例输入的维度需要与构图接口的 dims_size 匹配
                                           ));
