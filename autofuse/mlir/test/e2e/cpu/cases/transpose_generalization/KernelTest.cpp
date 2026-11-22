/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "test/e2e/cpu/cases/transpose_generalization/TransposeGeneralizationCase.h"
#include "autofuse_tiling_data.h"
#include "tikicpulib.h"
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0,
#if AF_TRANSPOSE_CASE == 2
                                                                    GM_ADDR input1,
#endif
                                                                    GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *, uint64_t *, uint32_t *);

namespace {
struct GmDeleter {
  void operator()(uint8_t *value) const {
    if (value) {
      AscendC::GmFree(value);
    }
  }
};
using Buffer = std::unique_ptr<uint8_t, GmDeleter>;
Buffer Allocate(uint64_t bytes) {
  return Buffer(static_cast<uint8_t *>(AscendC::GmAlloc(bytes ? bytes : 1)));
}
#if AF_TRANSPOSE_CASE == 1
using Input = half;
#else
using Input = float;
#endif

float InputValue(uint32_t index) {
  constexpr size_t kPatternPeriod = 127;
  constexpr int kPatternOffset = 63;
  return static_cast<float>(static_cast<int>(index % kPatternPeriod) - kPatternOffset) * 0.25F;
}

// Decode output coordinates independently of the generated tiling/kernel.
uint32_t SourceIndex(uint32_t index, const std::vector<int64_t> &input, const std::vector<int64_t> &output,
                     uint32_t reduceIndex = 0) {
#if AF_TRANSPOSE_CASE == 1
  const std::vector<size_t> perm = {1, 0};
#elif AF_TRANSPOSE_CASE == 3
  const std::vector<size_t> perm = {0, 3, 2, 1};
#else
  const std::vector<size_t> perm = {0, 2, 1};
#endif
  std::vector<uint32_t> coordinates(input.size(), 0);
  for (size_t axis = output.size(); axis > 0; --axis) {
    coordinates[perm[axis - 1]] = index % output[axis - 1];
    index /= output[axis - 1];
  }
#if AF_TRANSPOSE_CASE == 5
  coordinates[1] = reduceIndex;
#else
  (void)reduceIndex;
#endif
  uint32_t source = 0;
  for (size_t axis = 0; axis < input.size(); ++axis) {
    source = source * input[axis] + (input[axis] == 1 ? 0 : coordinates[axis]);
  }
  return source;
}
}  // namespace

TEST(MlirE2eCpu, TransposeGeneralization) {
  namespace cfg = mlir::afir::transpose_generalization;
  const auto in = cfg::GetShapeConfig(cfg::kInputShapes, 0);
  const auto out = cfg::GetShapeConfig(cfg::kOutputShapes, 0);
  const std::vector<int64_t> inputShape(in.shape), outputShape(out.shape);
  auto input = Allocate(in.element_count * sizeof(Input));
  auto output = Allocate(out.element_count * sizeof(float));
  ASSERT_TRUE(input);
  ASSERT_TRUE(output);
  for (uint32_t i = 0; i < in.element_count; ++i) {
    reinterpret_cast<Input *>(input.get())[i] = Input(InputValue(i));
  }
  for (uint32_t i = 0; i < out.element_count; ++i) {
    reinterpret_cast<float *>(output.get())[i] = std::numeric_limits<float>::quiet_NaN();
  }
#if AF_TRANSPOSE_CASE == 2
  auto other = Allocate(out.element_count * sizeof(float));
  ASSERT_TRUE(other);
  for (uint32_t i = 0; i < out.element_count; ++i) {
    reinterpret_cast<float *>(other.get())[i] = InputValue(i + 31);
  }
#endif
  AutofuseTilingData tiling{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  ASSERT_EQ(AutofuseE2eInitTiling(&tiling, &workspaceSize, &blockDim), 0);
  auto workspace = Allocate(workspaceSize);
  ASSERT_TRUE(workspace);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, input.get(),
#if AF_TRANSPOSE_CASE == 2
              other.get(),
#endif
              output.get(), workspace.get(), tiling);
  for (uint32_t i = 0; i < out.element_count; ++i) {
    float expected = 0;
#if AF_TRANSPOSE_CASE == 5
    for (uint32_t r = 0; r < static_cast<uint32_t>(inputShape[1]); ++r) {
      expected += InputValue(SourceIndex(i, inputShape, outputShape, r));
    }
#else
    expected = InputValue(SourceIndex(i, inputShape, outputShape));
#endif
#if AF_TRANSPOSE_CASE == 2
    expected += InputValue(i + 31);
#endif
    ASSERT_FLOAT_EQ(reinterpret_cast<float *>(output.get())[i], expected) << "output index=" << i;
  }
}
