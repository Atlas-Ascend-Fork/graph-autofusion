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
#include <limits>
#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"

extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling_data, uint64_t *workspace_size, uint32_t *block_dim,
                                  uint32_t aiv_num, uint32_t ub_size);
extern "C" void GetWorkspaceOffsets(const AutofuseTilingData *tiling_data, uint64_t *offsets);

class TestBackendWorkspaceUint64Tiling : public testing::Test {};

TEST_F(TestBackendWorkspaceUint64Tiling, ThreeLargeWorkspaceOffsetsWithoutAllocation) {
  constexpr uint64_t kWorkspaceBytes = 0x90000000ULL;
  constexpr uint64_t kReservedWorkspaceBytes = 16ULL * 1024ULL * 1024ULL;
  AutofuseTilingData tiling_data = {};
  uint64_t workspace_size = 0;
  uint32_t block_dim = 0;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48, 192 * 1024), 0);

  uint64_t workspace_offsets[3] = {};
  GetWorkspaceOffsets(&tiling_data, workspace_offsets);
  EXPECT_EQ(workspace_offsets[1] - workspace_offsets[0], kWorkspaceBytes);
  EXPECT_EQ(workspace_offsets[2] - workspace_offsets[1], kWorkspaceBytes);
  EXPECT_GE(workspace_offsets[2], 2 * kWorkspaceBytes);
  EXPECT_GT(workspace_offsets[2], std::numeric_limits<uint32_t>::max());
  EXPECT_GE(workspace_size, 3 * kWorkspaceBytes + kReservedWorkspaceBytes);
}
