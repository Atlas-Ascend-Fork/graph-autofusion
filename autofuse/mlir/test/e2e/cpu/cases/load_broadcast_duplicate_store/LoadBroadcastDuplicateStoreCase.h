/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_BROADCAST_DUPLICATE_STORE_CASE_H_
#define AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_BROADCAST_DUPLICATE_STORE_CASE_H_

#include "CaseConfig.h"

AF_MLIR_E2E_CPU_CASE_BEGIN(load_broadcast_duplicate_store, 1, 1);

// `Duplicate` 那一档的判据是「源在每根向量化轴上 repeat 都是 1」，也就是源只有一个元素。
static_assert(kInputShapes.begin()->element_count == 1U,
              "load_broadcast_duplicate_store requires a single-element source");

AF_MLIR_E2E_CPU_CASE_END(load_broadcast_duplicate_store)

#endif  // AUTOFUSE_MLIR_TEST_E2E_CPU_CASES_LOAD_BROADCAST_DUPLICATE_STORE_CASE_H_
