/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_abs_store/LoadAbsStoreCase.h"
#include "test/e2e/cpu/support/E2eTilingShape.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_abs_store::GetShapeConfig;
using mlir::afir::load_abs_store::kInputShapes;
using mlir::afir::load_abs_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadAbsStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  int32_t *input0 = static_cast<int32_t *>(AscendC::GmAlloc(input_element_count * sizeof(int32_t)));
  int32_t *output = static_cast<int32_t *>(AscendC::GmAlloc(output_element_count * sizeof(int32_t)));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  // Alternating signs so a lowering that dropped the sign handling entirely
  // would still have to get half the elements wrong to pass. int32 `Abs` is
  // synthesized by the wrapper out of Not/Adds/Max rather than issued as one
  // instruction, so getting the negative half right is the whole point.
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = (index % 2 == 0 ? 1 : -1) * static_cast<int32_t>(index % 97);
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);

  // Record which terms of the scheduled-transfer offset this run actually reaches.
  // ATT picks the split at run time, so without this the case can keep passing while
  // silently covering less than it was registered to cover.
  const mlir::afir::TilingShape tiling_shape = AF_E2E_TILING_SHAPE(tilingData, blockDim, output_element_count);
  mlir::afir::ReportTilingShape(AF_MLIR_E2E_CPU_CASE_NAME, tiling_shape);
  AF_E2E_ASSERT_TILING_COVERAGE(tiling_shape);
  uint8_t *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0 ? 1 : workspaceSize));
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<GM_ADDR>(input0),
              reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    EXPECT_EQ(output[index], std::abs(input0[index])) << "element " << index;
  }

  AscendC::GmFree(input0);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
