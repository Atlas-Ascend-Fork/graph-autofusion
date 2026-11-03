/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_SUM0_MUL_STORE_CASE_H_
#define AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_SUM0_MUL_STORE_CASE_H_

#include "CaseConfig.h"

#define AF_MLIR_E2E_CPU_STRINGIFY_IMPL(value) #value
#define AF_MLIR_E2E_CPU_STRINGIFY(value) AF_MLIR_E2E_CPU_STRINGIFY_IMPL(value)

namespace mlir::afir::load_sum0_mul_store {
using ::mlir::afir::GetShapeConfig;
using ::mlir::afir::test_config::kInputShapes;
using ::mlir::afir::test_config::kOutputShapes;

static_assert(kInputShapes.size() == 1, "load_sum0_mul_store requires exactly one input shape");
static_assert(kOutputShapes.size() == 1, "load_sum0_mul_store requires exactly one output shape");

constexpr const char *kKernelName = AF_MLIR_E2E_CPU_STRINGIFY(AF_MLIR_E2E_CPU_KERNEL_SYMBOL);
}  // namespace mlir::afir::load_sum0_mul_store

#endif  // AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_SUM0_MUL_STORE_CASE_H_
