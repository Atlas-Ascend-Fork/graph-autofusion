/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_unsigned_scalar_arithmetic_reg_base_store/UnsignedScalarArithmeticCase.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_unsigned_scalar_arithmetic_reg_base_store::Element;
using mlir::afir::load_unsigned_scalar_arithmetic_reg_base_store::GetShapeConfig;
using mlir::afir::load_unsigned_scalar_arithmetic_reg_base_store::kInputShapes;
using mlir::afir::load_unsigned_scalar_arithmetic_reg_base_store::kOutputShapes;
using mlir::afir::load_unsigned_scalar_arithmetic_reg_base_store::kScalar;
// 输出先填哨兵：它不等于任何一个期望值，kernel 漏写的元素不会被当成正确结果。
constexpr Element kSentinel = static_cast<Element>(0xA5U);
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR output, GM_ADDR workspace,
                                                                    AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, UnsignedScalarArithmeticPreservesUnsignedSemantics) {
  const auto release = [](void *buffer) { AscendC::GmFree(buffer); };
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  std::unique_ptr<Element[], decltype(release)> input0(
      static_cast<Element *>(AscendC::GmAlloc(input_element_count * sizeof(Element))), release);
  std::unique_ptr<Element[], decltype(release)> output(
      static_cast<Element *>(AscendC::GmAlloc(output_element_count * sizeof(Element))), release);
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(output, nullptr);

  constexpr Element max = static_cast<Element>(~Element{0});
  const Element values[] = {0, static_cast<Element>(kScalar - 1), kScalar, static_cast<Element>(kScalar + 1), max};
  for (uint32_t index = 0; index < input_element_count; ++index) {
    input0[index] = values[index % 5U];
  }
  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = kSentinel;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 1;
  ASSERT_EQ(AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim), 0);
  std::unique_ptr<uint8_t[], decltype(release)> workspace(
      static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize == 0 ? 1 : workspaceSize)), release);
  ASSERT_NE(workspace, nullptr);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<GM_ADDR>(input0.get()),
              reinterpret_cast<GM_ADDR>(output.get()), reinterpret_cast<GM_ADDR>(workspace.get()), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
#if defined(AFIR_SCALAR_Add)
    const Element expected = static_cast<Element>(static_cast<uint32_t>(input0[index]) + kScalar);
#elif defined(AFIR_SCALAR_Maximum)
    const Element expected = input0[index] > kScalar ? input0[index] : kScalar;
#else
    const Element expected = input0[index] < kScalar ? input0[index] : kScalar;
#endif
    EXPECT_EQ(output[index], expected) << "element " << index;
  }
}
