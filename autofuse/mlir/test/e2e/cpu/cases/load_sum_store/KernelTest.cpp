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
#include <limits>
#include <cstdint>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_sum_store/LoadSumStoreCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_sum_store::GetShapeConfig;
using mlir::afir::load_sum_store::kInputShapes;
using mlir::afir::load_sum_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

static void CheckReduceOutput(const float *input0, const float *output, const std::vector<int64_t> &input_shape,
                              uint32_t input_element_count, uint32_t output_element_count, bool reduces_leading_axis) {
  if (input_shape.size() == 3U) {
    // ARA：`[A0, R, A1]`，沿中间维求和。保留维有两根，所以 `A` 侧是连乘出来的——这正是
    // 这条注册存在的理由（见 `Case.cpp` 三维分支的注释）。
    const uint32_t leading = static_cast<uint32_t>(input_shape[0]);
    const uint32_t reduced = static_cast<uint32_t>(input_shape[1]);
    const uint32_t trailing = static_cast<uint32_t>(input_shape[2]);
    ASSERT_EQ(leading * reduced * trailing, input_element_count);
    ASSERT_EQ(output_element_count, leading * trailing);
    for (uint32_t outer = 0; outer < leading; ++outer) {
      for (uint32_t inner = 0; inner < trailing; ++inner) {
        float expected = 0.0F;
        for (uint32_t index = 0; index < reduced; ++index) {
          expected += input0[((outer * reduced) + index) * trailing + inner];
        }
        EXPECT_FLOAT_EQ(output[(outer * trailing) + inner], expected) << "at [" << outer << ", " << inner << "]";
      }
    }
  } else if (input_shape.size() == 2U) {
    const uint32_t rows = static_cast<uint32_t>(input_shape[0]);
    const uint32_t columns = static_cast<uint32_t>(input_shape[1]);
    ASSERT_EQ(rows * columns, input_element_count);
    // 归约哪一维由注册的输出形状决定，与 `Case.cpp` 建图时用的是同一条判据。
    if (reduces_leading_axis) {
      // RA：`[R, A]`，沿列求和。被归约维是 tile 的物理外维，每一列独立。
      ASSERT_EQ(output_element_count, columns);
      for (uint32_t column = 0; column < columns; ++column) {
        float expected = 0.0F;
        for (uint32_t row = 0; row < rows; ++row) {
          expected += input0[row * columns + column];
        }
        EXPECT_FLOAT_EQ(output[column], expected) << "column=" << column;
      }
    } else {
      // AR：`[A, R]`，沿行求和。
      ASSERT_EQ(output_element_count, rows);
      for (uint32_t row = 0; row < rows; ++row) {
        float expected = 0.0F;
        for (uint32_t column = 0; column < columns; ++column) {
          expected += input0[row * columns + column];
        }
#ifdef AF_SUM_MAX_TEST
        float maximum = std::numeric_limits<float>::lowest();
        for (uint32_t column = 0; column < columns; ++column) {
          maximum = std::max(maximum, input0[row * columns + column]);
        }
        expected += maximum;
#endif
        EXPECT_FLOAT_EQ(output[row], expected) << "row=" << row;
      }
    }
  } else {
    const auto [s0, s1, s2, s3] = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>{
        static_cast<uint32_t>(input_shape[0]), static_cast<uint32_t>(input_shape[1]),
        static_cast<uint32_t>(input_shape[2]), static_cast<uint32_t>(input_shape[3])};
    ASSERT_EQ(output_element_count, s0 * s2);
    for (uint32_t a0 = 0; a0 < s0; ++a0) {
      for (uint32_t a2 = 0; a2 < s2; ++a2) {
        float expected = 0.0F;
        for (uint32_t r1 = 0; r1 < s1; ++r1) {
          for (uint32_t r3 = 0; r3 < s3; ++r3) {
            const uint32_t input_index = ((a0 * s1 + r1) * s2 + a2) * s3 + r3;
            expected += input0[input_index];
          }
        }
        const uint32_t output_index = a0 * s2 + a2;
        EXPECT_FLOAT_EQ(output[output_index], expected) << "a0=" << a0 << ", a2=" << a2;
      }
    }
  }
}

#ifdef AF_RCORE_SCRATCH_TEST
static void CheckPartialWorkspace(const float *input0, void *workspace, const AutofuseTilingData &tilingData,
                                  bool reduces_leading_axis, uint32_t scratchRounds) {
  if (scratchRounds != 0) {
    const uint32_t r = scratchRounds == 2 ? 256U : 120U;
    const auto partials = tilingData.graph0_result1_g1_tiling_data.Rm_org_size;
    for (uint32_t part = 0; part < partials; ++part) {
      for (uint32_t a = 0; a < 8U; ++a) {
        float expected = 0.0F;
        for (uint32_t red = part * scratchRounds * r; red < (part + 1U) * scratchRounds * r && red < 1024U; ++red) {
          expected += input0[reduces_leading_axis ? red * 8U + a : a * 1024U + red];
        }
        EXPECT_FLOAT_EQ(static_cast<float *>(workspace)[part * 8U + a], expected)
            << "workspace part=" << part << ", A=" << a;
      }
    }
  }
}
#endif

#ifdef AF_RCORE_SCRATCH_TEST
static void ConfigureScratchRounds(AutofuseTilingData &tilingData, uint64_t &workspaceSize, uint32_t &blockDim,
                                   uint32_t scratchRounds) {
  if (scratchRounds != 0) {
    // Exercise repeated scratch reuse, including A/R tails, independently of ATT's preferred tile.
    const uint32_t a = scratchRounds == 2 ? 2U : 3U;
    const uint32_t r = scratchRounds == 2 ? 256U : 120U;
    const uint32_t partials = ((1024U + r - 1U) / r + scratchRounds - 1U) / scratchRounds;
    auto &first = tilingData.graph0_result1_g0_tiling_data;
#ifdef AF_RCORE_RA
    first.z1t_size = a;
    first.z0t_size = r;
    first.z0Tt_size = scratchRounds;
    first.q0_size = r * ((a + 7U) / 8U * 8U) * sizeof(float);
#else
    first.z0t_size = a;
    first.z1t_size = r;
    first.z1Tt_size = scratchRounds;
    first.q0_size = a * ((r + 7U) / 8U * 8U) * sizeof(float);
#endif
    first.q1_size = 32U;
    first.b0_size = first.b1_size = 8192U;
    first.block_dim = ((8U + a - 1U) / a) * partials;
    auto &second = tilingData.graph0_result1_g1_tiling_data;
    second.A_org_size = 8U;
    second.Rm_org_size = partials;
    second.At_size = 8U;
    second.Rmt_size = 1U;  // Reuse stage 2 scratch across all workspace rows too.
    second.ATb_size = 1U;
    second.q0_size = second.q1_size = 32U;
    second.b0_size = second.b1_size = 8192U;
    second.block_dim = 1U;
    tilingData.block_dim = blockDim = first.block_dim;
    workspaceSize = 512U;
  }
}
#endif

static void CheckLoadSumStoreResult(uint32_t scratchRounds = 0) {
  const auto &input_config = GetShapeConfig(kInputShapes, 0);
  const uint32_t input_element_count = input_config.element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  const std::vector<int64_t> input_shape(input_config.shape);
  const std::vector<int64_t> output_shape(GetShapeConfig(kOutputShapes, 0).shape);
  ASSERT_TRUE(input_shape.size() == 2U || input_shape.size() == 3U || input_shape.size() == 4U);
  const bool reduces_leading_axis = input_shape.size() == 2U && output_shape[0] == 1;

  float *input0 = static_cast<float *>(AscendC::GmAlloc(input_element_count * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(output_element_count * sizeof(float)));
  if (input0 == nullptr || output == nullptr) {
    if (input0 != nullptr) {
      AscendC::GmFree(input0);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  // Small integers keep every partial sum exactly representable in float.
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = static_cast<float>(index % 7U) - 3.0F;
#ifdef AF_SUM_MAX_TEST
    input0[index] -= static_cast<float>(index / input_shape[1]);
#endif
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0.0F;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(input0);
    AscendC::GmFree(output);
    FAIL() << "ATT tiling failed";
  }
#ifdef AF_SUM_MAX_TEST
  EXPECT_EQ(workspaceSize, 0U) << "the fused single-stage plan needs no GM workspace";
#endif
#ifdef AF_RCORE_SCRATCH_TEST
  ConfigureScratchRounds(tilingData, workspaceSize, blockDim, scratchRounds);
#endif
#ifdef AF_REDUCE_PARTITION
  ASSERT_GT(workspaceSize, 0U) << "partitioned reduction must use GM workspace";
#endif
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(input0);
      AscendC::GmFree(output);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(input0),
              reinterpret_cast<uint8_t *>(output), reinterpret_cast<uint8_t *>(workspace), tilingData);

  CheckReduceOutput(input0, output, input_shape, input_element_count, output_element_count, reduces_leading_axis);

#ifdef AF_RCORE_SCRATCH_TEST
  if (!::testing::Test::HasFatalFailure()) {
    CheckPartialWorkspace(input0, workspace, tilingData, reduces_leading_axis, scratchRounds);
  }
#endif

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}

TEST(MlirE2eCpu, LoadSumStoreCalculatesCorrectResult) {
  CheckLoadSumStoreResult();
}

#ifdef AF_RCORE_SCRATCH_TEST
TEST(MlirE2eCpu, RCoreScratchReusedAcrossTwoRounds) {
  CheckLoadSumStoreResult(2);
}

TEST(MlirE2eCpu, RCoreScratchReusedWithAxisTails) {
  CheckLoadSumStoreResult(3);
}
#endif
