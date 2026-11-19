/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Do not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>

#include "autofuse_tiling_data.h"

extern "C" int64_t AutofuseTiling(int64_t s0, int64_t s1, AutofuseTilingData *tiling_data, uint64_t *workspace_size,
                                  uint32_t *block_dim, uint32_t aiv_num, uint32_t ub_size);

namespace {
// 同分段（3公式布尔结果一致）但shape不同：第一次为算子级缓存与分段缓存双MISS，
// 全量搜索后首填分段缓存；第二次算子级缓存仍MISS（shape不同）、分段缓存HIT，
// 只对缓存的case求解。两次tiling key一致即证明同分段shape复用了同一历史最优模板选择。
int64_t RunSegmentTilingOnce(const int64_t s0, const int64_t s1, AutofuseTilingData *tiling_data) {
  AutofuseTilingData td = {};
  uint64_t workspace_size = 0;
  uint32_t block_dim = 0;
  const int64_t ret = AutofuseTiling(s0, s1, &td, &workspace_size, &block_dim, 48U, 192U * 1024U);
  *tiling_data = td;
  return ret;
}
}  // namespace

class TestBackendSegmentCacheTiling : public testing::Test {};

TEST_F(TestBackendSegmentCacheTiling, SegmentCacheHitReusesTemplateAcrossShapes) {
  AutofuseTilingData first = {};
  ASSERT_EQ(RunSegmentTilingOnce(4096, 128, &first), 0);

  AutofuseTilingData second = {};
  ASSERT_EQ(RunSegmentTilingOnce(4097, 128, &second), 0);

  // 同分段不同shape：分段缓存HIT后复用首次保存的模板选择，tiling key保持一致
  EXPECT_EQ(second.graph0_tiling_key, first.graph0_tiling_key);
  EXPECT_GT(second.get_block_dim(), 0U);
}
