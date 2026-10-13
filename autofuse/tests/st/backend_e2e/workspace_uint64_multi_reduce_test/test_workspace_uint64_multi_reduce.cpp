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
#include <gtest/gtest.h>

#include "tikicpulib.h"

#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void workspace_uint64_multi_reduce(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                                    GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling_data, uint64_t *workspace_size, uint32_t *block_dim,
                                  uint32_t aiv_num, uint32_t ub_size);

namespace {
constexpr uint64_t kWorkspaceBytes = 0x90000000ULL;
constexpr uint64_t kReservedWorkspaceBytes = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t kMinimumWorkspaceBytes = 3 * kWorkspaceBytes + kReservedWorkspaceBytes;
constexpr size_t kInputCount = 65536U;
constexpr size_t kOutputCount = 1024U;
constexpr float kExpectedOutput = 192.0F;
}  // namespace

class TestBackendWorkspaceUint64MultiReduceE2e : public testing::Test {};

TEST_F(TestBackendWorkspaceUint64MultiReduceE2e, ThreeWorkspacesUseUint64OffsetsAndReduceCorrectly) {
  AutofuseTilingData tiling_data = {};
  uint64_t workspace_size = 0;
  uint32_t block_dim = 0;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48, 192 * 1024), 0);

  EXPECT_EQ(tiling_data.get_workspace8() - tiling_data.get_workspace7(), kWorkspaceBytes);
  EXPECT_EQ(tiling_data.get_workspace9() - tiling_data.get_workspace8(), kWorkspaceBytes);
  EXPECT_GE(tiling_data.get_workspace9(), 2 * kWorkspaceBytes);
  EXPECT_GE(workspace_size, kMinimumWorkspaceBytes);

  float *input = static_cast<float *>(AscendC::GmAlloc(kInputCount * sizeof(float)));
  float *output = static_cast<float *>(AscendC::GmAlloc(kOutputCount * sizeof(float)));
  auto *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspace_size));
  if (input == nullptr || output == nullptr || workspace == nullptr) {
    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    GTEST_SKIP() << "Insufficient device memory for uint64 workspace E2E";
  }

  for (size_t i = 0; i < kInputCount; ++i) {
    input[i] = 1.0F;
  }
  for (size_t i = 0; i < kOutputCount; ++i) {
    output[i] = 0.0F;
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(workspace_uint64_multi_reduce, tiling_data.block_dim, reinterpret_cast<uint8_t *>(input),
              reinterpret_cast<uint8_t *>(output), workspace, reinterpret_cast<uint8_t *>(&tiling_data));

  for (size_t i = 0; i < kOutputCount; ++i) {
    EXPECT_EQ(output[i], kExpectedOutput);
  }

  AscendC::GmFree(input);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
}
