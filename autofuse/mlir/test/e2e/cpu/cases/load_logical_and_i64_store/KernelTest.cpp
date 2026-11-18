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
#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "test/e2e/cpu/cases/load_logical_and_i64_store/LoadLogicalAndI64StoreCase.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void AF_MLIR_E2E_CPU_KERNEL_SYMBOL(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                    GM_ADDR workspace, AutofuseTilingData tilingData);
extern "C" int64_t AutofuseE2eInitTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim);

namespace {
using mlir::afir::load_logical_and_i64_store::GetShapeConfig;
using mlir::afir::load_logical_and_i64_store::kInputShapes;
using mlir::afir::load_logical_and_i64_store::kOutputShapes;

// 分配与 tiling 的失败路径抽成 helper：TEST 主体的 nbnc 行数受 codecheck 超大函数
// 规则约束（<=50），三个 i64 e2e 用例共用同一段样板。
void *GmAllocChecked(size_t bytes, const char *what) {
  void *memory = AscendC::GmAlloc(bytes);
  if (memory == nullptr) {
    ADD_FAILURE() << "allocation failed: " << what;
  }
  return memory;
}

bool InitTilingAndWorkspace(AutofuseTilingData &tiling, uint32_t *blockDim, void **workspace) {
  uint64_t workspaceSize = 0;
  if (AutofuseE2eInitTiling(&tiling, &workspaceSize, blockDim) != 0) {
    ADD_FAILURE() << "ATT tiling failed";
    return false;
  }
  if (workspaceSize > 0U) {
    *workspace = AscendC::GmAlloc(workspaceSize);
    if (*workspace == nullptr) {
      ADD_FAILURE() << "workspace allocation failed";
      return false;
    }
  }
  return true;
}
}  // namespace

TEST(MlirE2eCpu, LoadLogicalAndI64StoreCalculatesCorrectResult) {
  const uint32_t input_element_count = GetShapeConfig(kInputShapes, 0).element_count;
  const uint32_t output_element_count = GetShapeConfig(kOutputShapes, 0).element_count;
  auto *lhs = static_cast<int64_t *>(GmAllocChecked(input_element_count * sizeof(int64_t), "lhs"));
  auto *rhs = static_cast<int64_t *>(GmAllocChecked(input_element_count * sizeof(int64_t), "rhs"));
  auto *output = static_cast<uint8_t *>(GmAllocChecked(output_element_count * sizeof(uint8_t), "output"));
  if (lhs == nullptr || rhs == nullptr || output == nullptr) {
    AscendC::GmFree(lhs);
    AscendC::GmFree(rhs);
    AscendC::GmFree(output);
    return;
  }

  // 逻辑族的输入域是 0 / 1，四个组合各占四分之一——AND 错生成其他逻辑 op 至少错四分之一。
  for (uint32_t index = 0; index < input_element_count; ++index) {
    lhs[index] = ((index % 4U) >= 2U) ? 1LL : 0LL;
    rhs[index] = ((index % 2U) == 1U) ? 1LL : 0LL;
  }
  std::fill_n(output, output_element_count, uint8_t{0});

  AutofuseTilingData tilingData{};
  uint32_t blockDim = 0;
  void *workspace = nullptr;
  if (!InitTilingAndWorkspace(tilingData, &blockDim, &workspace)) {
    AscendC::GmFree(lhs);
    AscendC::GmFree(rhs);
    AscendC::GmFree(output);
    return;
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AF_MLIR_E2E_CPU_KERNEL_SYMBOL, blockDim, reinterpret_cast<uint8_t *>(lhs),
              reinterpret_cast<uint8_t *>(rhs), reinterpret_cast<uint8_t *>(output),
              reinterpret_cast<uint8_t *>(workspace), tilingData);

  // 结果只有 0 / 1，用等号不用容差。
  for (uint32_t index = 0; index < output_element_count; ++index) {
    const auto expected = static_cast<uint8_t>((lhs[index] != 0LL && rhs[index] != 0LL) ? 1U : 0U);
    EXPECT_EQ(output[index], expected) << "index=" << index << " lhs=" << lhs[index] << " rhs=" << rhs[index];
  }

  AscendC::GmFree(lhs);
  AscendC::GmFree(rhs);
  AscendC::GmFree(output);
  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
