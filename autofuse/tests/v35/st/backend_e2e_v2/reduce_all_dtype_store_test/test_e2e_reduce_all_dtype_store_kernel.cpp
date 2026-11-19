/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint64_t *workspaceSize,
                                  uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size);
extern "C" __global__ __aicore__ void KERNEL_NAME(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class E2EBackendReduceAllDtypeStoreCode : public testing::Test {};

TEST_F(E2EBackendReduceAllDtypeStoreCode, CalculateCorrect) {
  constexpr int kRows = 2;
  constexpr int kCols = 16;
  constexpr int kInputSize = kRows * kCols;
  uint32_t block_dim = 48;
  AutofuseTilingData tiling_data;
  auto *input = static_cast<ELEMENT_TYPE *>(AscendC::GmAlloc(kInputSize * sizeof(ELEMENT_TYPE)));
  auto *output = static_cast<ELEMENT_TYPE *>(AscendC::GmAlloc(kRows * sizeof(ELEMENT_TYPE)));
  for (int i = 0; i < kInputSize; i++) {
    input[i] = static_cast<ELEMENT_TYPE>((i / kCols) == 0 ? 1 : ((i % kCols) == 0 ? 0 : 1));
  }
  for (int i = 0; i < kRows; i++) {
    output[i] = static_cast<ELEMENT_TYPE>(0);
  }

  uint64_t workspace_size = 0;
  ASSERT_EQ(AutofuseTiling(kRows, kCols, &tiling_data, &workspace_size, &block_dim, 48, 192 * 1024), 0);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(KERNEL_NAME, tiling_data.block_dim, reinterpret_cast<uint8_t *>(input),
              reinterpret_cast<uint8_t *>(output), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = 0;
  for (int i = 0; i < kRows; i++) {
    const ELEMENT_TYPE expected = static_cast<ELEMENT_TYPE>(i == 0 ? 1 : 0);
    if (output[i] != expected) {
      diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0U);
  AscendC::GmFree(input);
  AscendC::GmFree(output);
}
