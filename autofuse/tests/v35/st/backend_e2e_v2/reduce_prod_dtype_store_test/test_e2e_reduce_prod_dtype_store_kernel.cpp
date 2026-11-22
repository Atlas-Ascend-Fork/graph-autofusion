/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License"). You should not refer to the
 * License in the root of this software repository. This program is distributed in the hope that
 * it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the License in the root of this
 * software repository for specific language governing permissions and limitations under
 * the License.
 */

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint64_t *workspaceSize,
                                  uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size);
extern "C" __global__ __aicore__ void KERNEL_NAME(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class E2EBackendReduceProdDtypeStoreCode : public testing::Test {};

TEST_F(E2EBackendReduceProdDtypeStoreCode, CalculateCorrect) {
  constexpr int kRows = 2;
#ifndef REDUCE_TEST_COLS
#define REDUCE_TEST_COLS 1028
#endif
  constexpr int kCols = REDUCE_TEST_COLS;
  constexpr int kInputSize = kRows * kCols;
  uint32_t block_dim = 48;
  AutofuseTilingData tiling_data{};
  auto *input = static_cast<ELEMENT_TYPE *>(AscendC::GmAlloc(kInputSize * sizeof(ELEMENT_TYPE)));
  auto *output = static_cast<ELEMENT_TYPE *>(AscendC::GmAlloc(kRows * sizeof(ELEMENT_TYPE)));
  for (int i = 0; i < kInputSize; i++) {
    input[i] = static_cast<ELEMENT_TYPE>(1);
  }
  constexpr int kOverflowInputValue = sizeof(ELEMENT_TYPE) == 1 ? 4 : 100;
  input[0] = static_cast<ELEMENT_TYPE>(kOverflowInputValue);
  input[1] = static_cast<ELEMENT_TYPE>(kOverflowInputValue);
  input[kCols - 1] = static_cast<ELEMENT_TYPE>(-1);
  for (int i = 0; i < kRows; i++) {
    output[i] = static_cast<ELEMENT_TYPE>(0);
  }

  uint64_t workspace_size = 0;
  ASSERT_EQ(AutofuseTiling(kRows, kCols, &tiling_data, &workspace_size, &block_dim, 48, 192 * 1024), 0);
  auto *workspace = static_cast<uint8_t *>(workspace_size == 0 ? nullptr : AscendC::GmAlloc(workspace_size));
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(KERNEL_NAME, tiling_data.block_dim, reinterpret_cast<uint8_t *>(input),
              reinterpret_cast<uint8_t *>(output), workspace, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t diff_count = 0;
  ELEMENT_TYPE expected_values[kRows] = {static_cast<ELEMENT_TYPE>(-kOverflowInputValue * kOverflowInputValue),
                                         static_cast<ELEMENT_TYPE>(1)};
  for (int i = 0; i < kRows; i++) {
    if (output[i] != expected_values[i]) {
      diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0U);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
  AscendC::GmFree(input);
  AscendC::GmFree(output);
}
