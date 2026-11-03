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

#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_logical_or_store/Case.h"
#include "tikicpulib.h"

namespace {
using mlir::afir::load_logical_or_store::GetShapeConfig;
using mlir::afir::load_logical_or_store::kInputShapes;
using mlir::afir::load_logical_or_store::kOutputShapes;
}  // namespace

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

TEST(MlirE2eCpu, LoadLogicalOrStoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  // 输入与输出**不同宽**：这一族的结果恒为一个字节的真值。两边都按各自的元素宽度分配，
  // 照着输入的宽度分配输出会多要三倍的内存，照着输出的宽度分配输入会越界读。
  auto *lhs = static_cast<int32_t *>(AscendC::GmAlloc(input_element_count * sizeof(int32_t)));
  auto *rhs = static_cast<int32_t *>(AscendC::GmAlloc(input_element_count * sizeof(int32_t)));
  auto *output = static_cast<uint8_t *>(AscendC::GmAlloc(output_element_count * sizeof(uint8_t)));
  if (lhs == nullptr || rhs == nullptr || output == nullptr) {
    if (lhs != nullptr) {
      AscendC::GmFree(lhs);
    }
    if (rhs != nullptr) {
      AscendC::GmFree(rhs);
    }
    if (output != nullptr) {
      AscendC::GmFree(output);
    }
    FAIL() << "input/output allocation failed";
  }

  // **输入只喂 0 / 1，这不是图省事。**
  //
  // `logical.h` 里的 `LogicalOr` 不是一个会把输入归一化的逻辑运算：它把两个输入
  // `Cast` 成 half 之后直接取逐元素最大值，再 `Cast` 回 `uint8`。喂 2 和 5 进去，出来
  // 的是 5 而不是 1。**包装的前提是输入已经是 0 / 1 真值**——legacy 自己的 UT
  // （tests/ut/ascendc/api/test_logical.cpp）也是按 `std::rand() % 2` 喂的，期望值直接
  // 写成乘积（and）/ max（or）。
  //
  // 所以这份用例守的是同一个契约，下面那条断言才既是「逻辑或算对了」也是「包装算对
  // 了」——在 0 / 1 上两者重合。拿任意非零值去断言 1，是在测一条这个 API 从来没承诺过
  // 的性质。
  //
  // 四种组合都要有：(0,0) (0,1) (1,0) (1,1)。少喂一种的话，把 `||` 写成 `&&` 整套用例
  // 还是全绿。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    lhs[index] = ((index % 4U) >= 2U) ? 1 : 0;
    rhs[index] = ((index % 2U) == 1U) ? 1 : 0;
  }

  for (uint32_t index = 0; index < output_element_count; ++index) {
    output[index] = 0U;
  }

  AutofuseTilingData tilingData{};
  uint64_t workspaceSize = 0;
  uint32_t blockDim = 0;
  if (AutofuseE2eInitTiling(&tilingData, &workspaceSize, &blockDim) != 0) {
    AscendC::GmFree(lhs);
    AscendC::GmFree(rhs);
    AscendC::GmFree(output);
    FAIL() << "ATT tiling failed";
  }
  void *workspace = nullptr;
  if (workspaceSize > 0U) {
    workspace = AscendC::GmAlloc(workspaceSize);
    if (workspace == nullptr) {
      AscendC::GmFree(lhs);
      AscendC::GmFree(rhs);
      AscendC::GmFree(output);
      FAIL() << "workspace allocation failed";
    }
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(lhs),
              reinterpret_cast<uint8_t *>(rhs), reinterpret_cast<uint8_t *>(output),
              reinterpret_cast<uint8_t *>(workspace), tilingData);

  for (uint32_t index = 0; index < output_element_count; ++index) {
    // 输入是 0 / 1，结果也只有 0 / 1，所以这里用等号不用容差。
    const auto expected = static_cast<uint8_t>((lhs[index] != 0) || (rhs[index] != 0) ? 1U : 0U);
    EXPECT_EQ(output[index], expected) << "index=" << index << " lhs=" << lhs[index] << " rhs=" << rhs[index];
  }

  AscendC::GmFree(lhs);
  AscendC::GmFree(rhs);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
