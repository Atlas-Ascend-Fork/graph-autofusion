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
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include "autofuse_tiling_data.h"
#include "Case.h"
#include "tikicpulib.h"

namespace {
float Reference(float lhs, float rhs) {
  switch (AF_BRC_KIND) {
    case 0:
      return lhs + rhs;
    case 1:
      return lhs * rhs;
    case 2:
      return std::min(lhs, rhs);
    case 3:
      return std::max(lhs, rhs);
    case 4:
      return lhs - rhs;
    default:
      if (rhs == 0.0F) {
        ADD_FAILURE() << "Reference division requires a nonzero rhs";
        return std::numeric_limits<float>::quiet_NaN();
      }
      return lhs / rhs;
  }
}
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *, uint64_t *, uint32_t *);

TEST(MlirE2eCpu, BinaryBroadcastInlineRegBase) {
  namespace cfg = mlir::afir::binary_brc_inline_reg_base;
  const auto lhsShape = cfg::GetShapeConfig(cfg::kInputShapes, 0);
  const auto rhsShape = cfg::GetShapeConfig(cfg::kInputShapes, 1);
  const auto outShape = cfg::GetShapeConfig(cfg::kOutputShapes, 0);
  auto *lhs = static_cast<float *>(AscendC::GmAlloc(lhsShape.element_count * sizeof(float)));
  auto *rhs = static_cast<float *>(AscendC::GmAlloc(rhsShape.element_count * sizeof(float)));
  auto *out = static_cast<float *>(AscendC::GmAlloc(outShape.element_count * sizeof(float)));
  ASSERT_NE(lhs, nullptr);
  ASSERT_NE(rhs, nullptr);
  ASSERT_NE(out, nullptr);
  // Nonzero, asymmetric signed operands expose reversed Sub/Div and stale rows.
  for (uint32_t i = 0; i < lhsShape.element_count; ++i) {
    lhs[i] = static_cast<float>(static_cast<int>(i % 29U) - 14) + 0.5F;
  }
  for (uint32_t i = 0; i < rhsShape.element_count; ++i) {
    rhs[i] = static_cast<float>(static_cast<int>(i % 17U) - 8) + 0.25F;
  }
  std::fill_n(out, outShape.element_count, -9999.0F);
  AutofuseTilingData tiling{};
  uint64_t workspaceSize = 0;
  uint32_t blocks = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tiling, &workspaceSize, &blocks), 0);
  // Execute the outer-row VF template, not just a kernel containing that branch.
  ASSERT_EQ(tiling.get_tiling_key(), 0U);
  ASSERT_GT(tiling.get_z0t_size(), 1U);
  auto *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0 ? 1 : workspaceSize));
  ASSERT_NE(workspace, nullptr);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blocks, reinterpret_cast<GM_ADDR>(lhs), reinterpret_cast<GM_ADDR>(rhs),
              reinterpret_cast<GM_ADDR>(out), reinterpret_cast<GM_ADDR>(workspace), tiling);
  for (uint32_t i = 0; i < outShape.element_count; ++i) {
    const auto l = mlir::afir::BroadcastSourceIndex(outShape.shape, lhsShape.shape, i);
    const auto r = mlir::afir::BroadcastSourceIndex(outShape.shape, rhsShape.shape, i);
    EXPECT_NEAR(out[i], Reference(lhs[l], rhs[r]), 1.0e-4F) << "element " << i;
  }
  AscendC::GmFree(lhs);
  AscendC::GmFree(rhs);
  AscendC::GmFree(out);
  AscendC::GmFree(workspace);
}
