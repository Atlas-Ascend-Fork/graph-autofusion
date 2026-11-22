/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

// 本用例的优化阶段(含 SwapCastBroadcastPass 的 load-cast-brc 交换)验证在生成器
// load_cast_brc_realign_backend_generate 中完成; 本文件仅为 backend_e2e_st_test 宏的
// TEST_SRC 槽位提供可编译、恒通过的用例占位。
TEST(LoadCastBrcRealign, OptimizeStageCoveredByGenerator) {
  SUCCEED();
}
