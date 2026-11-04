/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_AXPY_HALF_STORE_CASE_H_
#define AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_AXPY_HALF_STORE_CASE_H_

#include "CaseConfig.h"

AF_MLIR_E2E_CPU_CASE_BEGIN(load_axpy_half_store, 2, 1);

// 系数。用 2 的负幂，f16 里也精确可表示，`x1 + alpha * x2` 的期望值不引入舍入。
//
// **alpha 恒是 f32，不跟 tile 的元素类型走**：`AxpyExtend` 的形参写死 `const float
// alpha`（不是 `const T&`），这一点与 `afir.leaky_relu` 恰好相反。f16 这份用例顺带把
// 这条也钉住了——常量要是跟着 tile 建成了 half，生成的 kernel 里就不是 `constexpr
// float`。
constexpr float kAlpha = 0.5F;

AF_MLIR_E2E_CPU_CASE_END(load_axpy_half_store)

#endif  // AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_AXPY_HALF_STORE_CASE_H_
