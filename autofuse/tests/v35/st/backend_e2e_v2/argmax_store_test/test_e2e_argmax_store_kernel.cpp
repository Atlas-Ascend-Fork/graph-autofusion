/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void argmax_store_test(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
#if defined(ARGMAX_SINGLE_ROW) || defined(ARGMAX_ARA)
#ifdef ARGMAX_ARA
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint64_t *workspace_size, uint32_t *block_dim,
                                  uint32_t aiv_num, uint32_t ub_size);
#else
extern "C" int64_t AutofuseTiling(uint32_t s1, AutofuseTilingData *tiling, uint64_t *workspace_size,
                                  uint32_t *block_dim, uint32_t aiv_num, uint32_t ub_size);
#endif
#else
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData *tiling, uint64_t *workspace_size,
                                  uint32_t *block_dim, uint32_t aiv_num, uint32_t ub_size);
#endif

namespace {
void RunArgmax(bool all_nan) {
#ifdef ARGMAX_ARA
  constexpr uint32_t dim0 = 4;
  constexpr uint32_t dim1 = 8;
  constexpr uint32_t dim2 = 2048;
  constexpr uint32_t reduce = 32;
  constexpr uint32_t dim4 = 16;
  constexpr uint64_t input_elements = static_cast<uint64_t>(dim0) * dim1 * dim2 * reduce * dim4;
  constexpr uint64_t output_elements = static_cast<uint64_t>(dim0) * dim1 * dim2 * dim4;
#elif defined(ARGMAX_SINGLE_ROW)
  constexpr uint32_t rows = 1;
#else
  constexpr uint32_t rows = 8;
#endif
#ifndef ARGMAX_ARA
  constexpr uint32_t columns = 4096;
#endif
  AutofuseTilingData tiling{};
  uint64_t workspace_size = 0;
  uint32_t block_dim = 0;
#ifdef ARGMAX_ARA
  std::vector<int64_t> expected(output_elements, 0);
  ASSERT_EQ(AutofuseTiling(&tiling, &workspace_size, &block_dim, 48, 192 * 1024), 0);
#elif defined(ARGMAX_SINGLE_ROW)
  ASSERT_EQ(AutofuseTiling(columns, &tiling, &workspace_size, &block_dim, 48, 192 * 1024), 0);
#else
  ASSERT_EQ(AutofuseTiling(rows, columns, &tiling, &workspace_size, &block_dim, 48, 192 * 1024), 0);
#endif
  ASSERT_GT(block_dim, 1U);
  ASSERT_GT(workspace_size, 0U);

  const auto free_gm = [](void *ptr) { AscendC::GmFree(ptr); };
#ifdef ARGMAX_ARA
  std::unique_ptr<void, decltype(free_gm)> input(AscendC::GmAlloc(input_elements * sizeof(float)), free_gm);
  std::unique_ptr<void, decltype(free_gm)> output(AscendC::GmAlloc(output_elements * sizeof(int64_t)), free_gm);
#else
  std::unique_ptr<void, decltype(free_gm)> input(AscendC::GmAlloc(rows * columns * sizeof(float)), free_gm);
  std::unique_ptr<void, decltype(free_gm)> output(AscendC::GmAlloc(rows * sizeof(int64_t)), free_gm);
#endif
  std::unique_ptr<void, decltype(free_gm)> workspace(AscendC::GmAlloc(workspace_size), free_gm);
  ASSERT_TRUE(input && output && workspace);
  auto *indices = static_cast<int64_t *>(output.get());
  auto *values = static_cast<float *>(input.get());
#ifdef ARGMAX_ARA
  for (uint32_t d0 = 0; d0 < dim0; ++d0) {
    for (uint32_t d1 = 0; d1 < dim1; ++d1) {
      for (uint32_t d2 = 0; d2 < dim2; ++d2) {
        for (uint32_t d3 = 0; d3 < reduce; ++d3) {
          for (uint32_t d4 = 0; d4 < dim4; ++d4) {
            const uint64_t index = (((static_cast<uint64_t>(d0) * dim1 + d1) * dim2 + d2) * reduce + d3) * dim4 + d4;
            const uint64_t output_index = ((static_cast<uint64_t>(d0) * dim1 + d1) * dim2 + d2) * dim4 + d4;
            const uint32_t winner = (output_index * 13 + output_index / dim4) % reduce;
            float current = d3 == winner || d3 == reduce - 1 ? 100.0F : -static_cast<float>(d3);
            if (all_nan || (output_index % 7 == 0 && (d3 == winner || d3 == reduce - 1))) {
              current = std::numeric_limits<float>::quiet_NaN();
            }
            values[index] = current;
            if (d3 != 0) {
              const uint64_t best_offset = index - (d3 - expected[output_index]) * dim4;
              const float best = values[best_offset];
              if (!std::isnan(best) && (std::isnan(current) || current > best)) {
                expected[output_index] = d3;
              }
            }
          }
        }
      }
    }
  }
#else
  for (uint32_t i = 0; i < rows * columns; ++i) {
    values[i] = all_nan ? std::numeric_limits<float>::quiet_NaN() : static_cast<float>(i % columns);
  }
#endif
  // Poison unwritten workspace: incorrect Phase2 strides must not pass the finite-input case.
  std::fill_n(static_cast<uint8_t *>(workspace.get()), workspace_size, 0xFF);
#ifdef ARGMAX_ARA
  std::fill_n(indices, output_elements, -1);
#else
  std::fill_n(indices, rows, -1);
#endif

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(argmax_store_test, block_dim, static_cast<uint8_t *>(input.get()), static_cast<uint8_t *>(output.get()),
              static_cast<uint8_t *>(workspace.get()), reinterpret_cast<uint8_t *>(&tiling));

#ifdef ARGMAX_ARA
  for (uint64_t i = 0; i < output_elements; ++i) {
    EXPECT_EQ(indices[i], expected[i]) << "output " << i;
  }
#else
  for (uint32_t row = 0; row < rows; ++row) {
    EXPECT_EQ(indices[row], all_nan ? 0 : columns - 1) << "row " << row;
  }
#endif
}
}  // namespace

#ifdef ARGMAX_ARA
TEST(E2EBackendArgmaxStoreCode, AllNan) {
#else
TEST(E2EBackendArgmaxStoreCode, AllNanRCore) {
#endif
  RunArgmax(true);
}

#ifdef ARGMAX_ARA
TEST(E2EBackendArgmaxStoreCode, Finite) {
#else
TEST(E2EBackendArgmaxStoreCode, FiniteRCore) {
#endif
  RunArgmax(false);
}
