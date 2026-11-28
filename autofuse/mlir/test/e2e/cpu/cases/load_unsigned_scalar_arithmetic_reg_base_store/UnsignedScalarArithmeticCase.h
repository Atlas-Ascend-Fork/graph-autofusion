/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_UNSIGNED_SCALAR_ARITHMETIC_CASE_H
#define AUTOFUSE_MLIR_UNSIGNED_SCALAR_ARITHMETIC_CASE_H

#include <cstdint>
#include <type_traits>

#include "CaseConfig.h"

AF_MLIR_E2E_CPU_CASE_BEGIN(load_unsigned_scalar_arithmetic_reg_base_store, 1, 1);
// 两种宽度的分界：`AFIR_SCALAR_WIDTH` 等于它是 ui16，否则是 ui32（CMakeLists.txt 只生成这两种）。
inline constexpr int kUint16Width = 16;
using Element = std::conditional_t<AFIR_SCALAR_WIDTH == kUint16Width, uint16_t, uint32_t>;
// 加 / 取大 / 取小三个算子共用的标量操作数：最高位置 1，再添一个个位数偏移。它落在该位宽无符号
// 范围的上半段（对同宽的有符号类型是负数），又不是 2 的幂；`kScalar ± 1`、`kScalar` 与最大值
// 都在测试数据里。
inline constexpr uint32_t kScalarLowBits = 7U;
inline constexpr Element kScalar = static_cast<Element>((uint32_t{1} << (AFIR_SCALAR_WIDTH - 1U)) + kScalarLowBits);
AF_MLIR_E2E_CPU_CASE_END(load_unsigned_scalar_arithmetic_reg_base_store)

#endif
