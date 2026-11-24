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

#ifndef REDUCE_TEST_ROWS
#define REDUCE_TEST_ROWS 2
#endif
#ifndef REDUCE_TEST_COLS
#define REDUCE_TEST_COLS 1028
#endif
#ifndef REDUCE_PARTIAL_MASK_CASE
#define REDUCE_PARTIAL_MASK_CASE 0
#endif
namespace {
constexpr int kRows = REDUCE_TEST_ROWS;
constexpr int kCols = REDUCE_TEST_COLS;
constexpr int kInputSize = kRows * kCols;
constexpr int kPartialMaskRows = 4;
constexpr int kPartialMaskCols = 3;
constexpr int kPartialMaskZero = 0;
constexpr int kPartialMaskOne = 1;
constexpr int kPartialMaskTwo = 2;
constexpr int kPartialMaskThree = 3;
constexpr int kPartialMaskSix = 6;

void InitializePartialMaskInput(ELEMENT_TYPE *input) {
  const ELEMENT_TYPE partial_mask_inputs[kPartialMaskRows][kPartialMaskCols] = {
      {kPartialMaskZero, kPartialMaskZero, kPartialMaskZero},
      {kPartialMaskOne, kPartialMaskOne, kPartialMaskOne},
      {kPartialMaskZero, kPartialMaskTwo, kPartialMaskThree},
      {kPartialMaskThree, kPartialMaskTwo, kPartialMaskOne}};
  for (int row = 0; row < kRows; row++) {
    for (int col = 0; col < kCols; col++) {
      input[row * kCols + col] = partial_mask_inputs[row][col];
    }
  }
}

void InitializeDefaultInput(ELEMENT_TYPE *input) {
  for (int i = 0; i < kInputSize; i++) {
    input[i] = static_cast<ELEMENT_TYPE>(1);
  }
  constexpr int kOverflowInputValue = sizeof(ELEMENT_TYPE) == 1 ? 4 : 100;
  input[0] = static_cast<ELEMENT_TYPE>(kOverflowInputValue);
  input[1] = static_cast<ELEMENT_TYPE>(kOverflowInputValue);
  input[kCols - 1] = static_cast<ELEMENT_TYPE>(-1);
}

void InitializeExpectedValues(ELEMENT_TYPE *expected_values) {
  if constexpr (REDUCE_PARTIAL_MASK_CASE) {
    expected_values[0] = kPartialMaskZero;
    expected_values[1] = kPartialMaskOne;
    expected_values[2] = kPartialMaskZero;
    expected_values[3] = kPartialMaskSix;
  } else {
    constexpr int kOverflowInputValue = sizeof(ELEMENT_TYPE) == 1 ? 4 : 100;
    expected_values[0] = static_cast<ELEMENT_TYPE>(-kOverflowInputValue * kOverflowInputValue);
    expected_values[1] = 1;
  }
}
}  // namespace

TEST_F(E2EBackendReduceProdDtypeStoreCode, CalculateCorrect) {
  uint32_t block_dim = 48;
  AutofuseTilingData tiling_data{};
  auto *input = static_cast<ELEMENT_TYPE *>(AscendC::GmAlloc(kInputSize * sizeof(ELEMENT_TYPE)));
  auto *output = static_cast<ELEMENT_TYPE *>(AscendC::GmAlloc(kRows * sizeof(ELEMENT_TYPE)));
  if constexpr (REDUCE_PARTIAL_MASK_CASE) {
    InitializePartialMaskInput(input);
  } else {
    InitializeDefaultInput(input);
  }
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
  ELEMENT_TYPE expected_values[kRows] = {};
  InitializeExpectedValues(expected_values);
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
