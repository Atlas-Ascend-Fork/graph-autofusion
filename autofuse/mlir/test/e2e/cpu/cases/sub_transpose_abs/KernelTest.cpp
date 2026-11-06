/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/sub_transpose_abs/SubTransposeAbsCase.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *, uint64_t *, uint32_t *);

namespace {
struct GmDeleter {
  void operator()(float *p) const {
    if (p) {
      AscendC::GmFree(p);
    }
  }
};
using GmBuffer = std::unique_ptr<float, GmDeleter>;
GmBuffer Allocate(uint64_t bytes) {
  return GmBuffer(static_cast<float *>(AscendC::GmAlloc(bytes == 0 ? 1 : bytes)));
}
}  // namespace

TEST(MlirE2eCpu, SubTransposeAbs) {
  namespace config = mlir::afir::sub_transpose_abs;
  const auto shape = config::GetShapeConfig(config::kInputShapes, 0);
  const std::vector<int64_t> dims(shape.shape);
  auto x = Allocate(shape.element_count * sizeof(float));
  auto y = Allocate(shape.element_count * sizeof(float));
  auto out = Allocate(shape.element_count * sizeof(float));
  ASSERT_TRUE(x);
  ASSERT_TRUE(y);
  ASSERT_TRUE(out);
  for (uint32_t i = 0; i < shape.element_count; ++i) {
    x.get()[i] = static_cast<float>(static_cast<int>(i % 127) - 63) * 0.25F;
    y.get()[i] = static_cast<float>(static_cast<int>(i % 41) - 20) * 0.5F;
    out.get()[i] = NAN;
  }
  AutofuseTilingData tiling{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  ASSERT_EQ(AutofuseE2eInitTiling(&tiling, &workspaceSize, &blockDim), 0);
  auto workspace = Allocate(workspaceSize);
  ASSERT_TRUE(workspace);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(x.get()),
              reinterpret_cast<uint8_t *>(y.get()), reinterpret_cast<uint8_t *>(out.get()),
              reinterpret_cast<uint8_t *>(workspace.get()), tiling);
  for (uint32_t i = 0; i < shape.element_count; ++i) {
    const uint32_t w = i % dims[2];
    // Output [B, A, C] maps (b, a, c) back to input (a, b, c).
    const uint32_t a = (i / dims[2]) % dims[0];
    const uint32_t b = i / (dims[0] * dims[2]);
    const uint32_t source = (a * dims[1] + b) * dims[2] + w;
    EXPECT_FLOAT_EQ(out.get()[i], std::fabs(x.get()[source] - y.get()[source])) << i;
  }
}
