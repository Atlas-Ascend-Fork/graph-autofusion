/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_MEAN_STORE_CASE_H_
#define AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_MEAN_STORE_CASE_H_

#include "CaseConfig.h"

#define AF_MLIR_E2E_CPU_STRINGIFY_IMPL(value) #value
#define AF_MLIR_E2E_CPU_STRINGIFY(value) AF_MLIR_E2E_CPU_STRINGIFY_IMPL(value)

namespace mlir::afir::load_mean_store {
using ::mlir::afir::GetShapeConfig;
using ::mlir::afir::test_config::kInputShapes;
using ::mlir::afir::test_config::kOutputShapes;

static_assert(kInputShapes.size() == 1, "load_mean_store requires exactly one input shape");
static_assert(kOutputShapes.size() == 1, "load_mean_store requires exactly one output shape");

// 与 `load_sum_store` 同构，只把算子换成 `Mean`：归约轴的选法、输出形状的约定都一样
// （`[M, 1]` 归约末维走 AR，`[1, N]` 归约首维走 RA）。
//
// 这组用例验的是 `Mean` 多出来的那一次缩放，判据有三条，缺一条就有一类错漏过去：
//   - 分母是完整归约域的元素数，不是某一轮的；
//   - 每个输出槽位恰好除一次，跨轮时不是每轮都除；
//   - 32B 对齐补出来的位置不计入分母（`16x37` 那条的分母必须是 37，不是 40）。
// 对应的三类错分别是「均值的均值」、「重复除成 1/N^rounds」和「分母偏大、结果偏小」。
constexpr const char *kKernelName = AF_MLIR_E2E_CPU_STRINGIFY(AF_MLIR_E2E_CPU_KERNEL_SYMBOL);
}  // namespace mlir::afir::load_mean_store

#endif  // AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_MEAN_STORE_CASE_H_
