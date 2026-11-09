/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the License). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void floor_div_int64_store_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR workspace,
                                                                 GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint32_t *workspaceSize,
                                  uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2EBackendFloorDivInt64StoreCode : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

static int64_t FloorDivInt64Reference(int64_t lhs, int64_t rhs) {
  if (rhs == 0) {
    return 0;
  }
  int64_t quotient = lhs / rhs;
  const int64_t remainder = lhs % rhs;
  if (remainder != 0 && ((remainder > 0) != (rhs > 0))) {
    quotient--;
  }
  return quotient;
}

TEST_P(E2EBackendFloorDivInt64StoreCode, CalculateCorrect) {
  auto floor_div_shape = GetParam();
  uint64_t floor_div_block_dim = 48;
  int floor_div_test_size = floor_div_shape[0] * floor_div_shape[1];

  AutofuseTilingData tiling_data;
  int64_t *floor_div_x1 = static_cast<int64_t *>(AscendC::GmAlloc(floor_div_test_size * sizeof(int64_t) + 32));
  int64_t *floor_div_x2 = static_cast<int64_t *>(AscendC::GmAlloc(floor_div_test_size * sizeof(int64_t) + 32));
  int64_t *floor_div_y = static_cast<int64_t *>(AscendC::GmAlloc(floor_div_test_size * sizeof(int64_t) + 32));
  int64_t *floor_div_expect = static_cast<int64_t *>(AscendC::GmAlloc(floor_div_test_size * sizeof(int64_t) + 32));

  for (int i = 0; i < floor_div_test_size; i++) {
    floor_div_x1[i] = static_cast<int64_t>((i % 31) - 15);
    floor_div_x2[i] = static_cast<int64_t>((i % 7) + 1);
    if ((i % 5) == 0) {
      floor_div_x2[i] = -floor_div_x2[i];
    }
    floor_div_expect[i] = FloorDivInt64Reference(floor_div_x1[i], floor_div_x2[i]);
  }

  uint32_t ws_size = 0;
  AutofuseTiling(floor_div_shape[0], floor_div_shape[1], &tiling_data, &ws_size, &floor_div_block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(floor_div_int64_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(floor_div_x1),
              reinterpret_cast<uint8_t *>(floor_div_x2), reinterpret_cast<uint8_t *>(floor_div_y), nullptr,
              reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t floor_div_diff_count = 0;
  for (int i = 0; i < floor_div_test_size; i++) {
    if (floor_div_y[i] != floor_div_expect[i]) {
      floor_div_diff_count++;
    }
  }
  EXPECT_EQ(floor_div_diff_count, 0U) << " of " << floor_div_test_size;

  AscendC::GmFree(floor_div_x1);
  AscendC::GmFree(floor_div_x2);
  AscendC::GmFree(floor_div_y);
  AscendC::GmFree(floor_div_expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2EBackendFloorDivInt64StoreCode,
                         ::testing::Values(std::vector<int>{32, 16}, std::vector<int>{32, 18},
                                           std::vector<int>{512, 15}));
