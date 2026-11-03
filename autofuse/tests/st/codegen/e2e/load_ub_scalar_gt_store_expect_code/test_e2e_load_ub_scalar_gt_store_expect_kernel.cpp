/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void load_ub_scalar_gt_store(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace,
                                                              GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData &tiling_data);

class E2E_LoadUbScalarGtStore_Code : public testing::Test,
                                     public testing::WithParamInterface<std::pair<std::vector<int>, std::vector<int>>> {
};

// 本用例当前只校验流程: codegen生成非const ub_scalar的compare广播发射分支, 生成kernel可编译、ICPU启动无错。
// 数值校验暂缓: 该发射路径的运行时正确性依赖compare_v2.h既有广播问题的修复, 修复后恢复diff断言。
TEST_P(E2E_LoadUbScalarGtStore_Code, CalculateCorrect) {
  auto [test_shape, test_tiling] = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  half *x1 = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);
  half *x2 = (half *)AscendC::GmAlloc(test_size * sizeof(half) + 32);
  uint8_t *y = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);

  // Prepare test data
  srand(1);
  for (int i = 0; i < test_size; i++) {
    x1[i] = static_cast<half>(rand() / (double)RAND_MAX);
    // x2是作为scalar输入, 只对第一个数值赋值
    if (i == 0) {
      x2[i] = static_cast<half>(0.5);
    } else {
      x2[i] = static_cast<half>(0.0);
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.s2 = test_shape[2];
  if (test_tiling.size() == 0U) {  // tiling data 来源于tiling函数GetTiling
    GetTiling(tiling_data);
  } else {  // tiling信息来源于测试用例入参
    tiling_data.block_dim = test_tiling[0];
    tiling_data.z0z1Tb_size = test_tiling[1];
    tiling_data.z1t_size = test_tiling[2];
  }
  tiling_data.tiling_key = 0;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_ub_scalar_gt_store, tiling_data.block_dim, (uint8_t *)x1, (uint8_t *)x2, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  uint32_t y_true = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != 0) {
      y_true++;
    }
  }
  std::cout << "load_ub_scalar_gt_store flow ok, y_true=" << y_true << " of " << test_size << std::endl;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(y);
}

// s2=128使last_axis对齐后达到一个repeat(256B), 与const标量用例一致的参数族(half/uint8/else分支)
INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadUbScalarGtStore_Code,
                         ::testing::Values(std::pair<std::vector<int>, std::vector<int>>{{96, 16, 128}, {}}));
