/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_ADD_SCALAR_DATA_STORE_CASE_H_
#define AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_ADD_SCALAR_DATA_STORE_CASE_H_

#include "CaseConfig.h"

AF_MLIR_E2E_CPU_CASE_BEGIN(load_add_scalar_data_store, 1, 1);

// 只有 Data 那一个输入有形状。`ScalarData` 是 index 1 的输入，但它按值传、没有形状，
// 所以不进 INPUT_SHAPES。

AF_MLIR_E2E_CPU_CASE_END(load_add_scalar_data_store)

#endif  // AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_ADD_SCALAR_DATA_STORE_CASE_H_
