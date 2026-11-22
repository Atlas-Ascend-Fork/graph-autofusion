/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_AXPY_STORE_CASE_H_
#define AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_AXPY_STORE_CASE_H_

#include "CaseConfig.h"

AF_MLIR_E2E_CPU_CASE_BEGIN(load_axpy_store, 2, 1);

// 系数使用 2 的负幂，可被 f32 精确表示，避免 x1 + alpha*x2 的参考值引入额外舍入。
// 与 load_leaky_relu_store 的斜率选择一致。
constexpr float kAlpha = 0.5F;

AF_MLIR_E2E_CPU_CASE_END(load_axpy_store)

#endif  // AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_AXPY_STORE_CASE_H_
