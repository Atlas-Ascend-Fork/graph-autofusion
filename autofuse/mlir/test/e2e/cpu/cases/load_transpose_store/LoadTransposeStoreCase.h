/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_TRANSPOSE_STORE_CASE_H_
#define AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_TRANSPOSE_STORE_CASE_H_

#include "CaseConfig.h"

#define AF_MLIR_E2E_CPU_STRINGIFY_IMPL(value) #value
#define AF_MLIR_E2E_CPU_STRINGIFY(value) AF_MLIR_E2E_CPU_STRINGIFY_IMPL(value)

namespace mlir::afir::load_transpose_store {
using ::mlir::afir::GetShapeConfig;
using ::mlir::afir::test_config::kInputShapes;
using ::mlir::afir::test_config::kOutputShapes;

static_assert(kInputShapes.size() == 1, "load_transpose_store requires exactly one input shape");
static_assert(kOutputShapes.size() == 1, "load_transpose_store requires exactly one output shape");
constexpr const char *kKernelName = AF_MLIR_E2E_CPU_STRINGIFY(AF_MLIR_E2E_CPU_KERNEL_SYMBOL);
}  // namespace mlir::afir::load_transpose_store

#endif  // AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_TRANSPOSE_STORE_CASE_H_
