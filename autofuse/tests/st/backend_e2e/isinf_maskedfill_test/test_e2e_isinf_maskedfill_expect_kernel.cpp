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
#include <vector>
#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void isinf_maskedfill_test(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData *tiling,
                                  uint64_t *workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendIsInfMaskedFill_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_BackendIsInfMaskedFill_Code, CalculateCorrect) {
  // CPU simulator has known issues with ReinterpretCast aliasing for Abs operation.
  // Keep this target as compile-only coverage for generated IsInf/MaskedFill kernel code.
  GTEST_SKIP() << "CPU simulator ReinterpretCast aliasing issue; compile-only coverage";
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendIsInfMaskedFill_Code,
                         ::testing::Values(std::vector<int>{32, 16, 16}, std::vector<int>{32, 16, 18},
                                           std::vector<int>{32, 512, 15}));
