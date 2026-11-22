/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Included inside the owning solver namespace; dependencies are supplied by its translation unit.
// AF_AXES_RESOURCE_BEGIN
template <typename T, typename U> inline auto Max(T a, U b) { return static_cast<double>(a) > static_cast<double>(b) ? a : b; }
template <typename T, typename U> inline auto Min(T a, U b) { return static_cast<double>(a) < static_cast<double>(b) ? a : b; }
template <typename T> inline auto Abs(T a) { return static_cast<double>(a) >= 0 ? a : -a; }
#define Log(a) (log((double)(a)))
#define Pow(a, b) pow(a, b)
#define Rational(a, b) ((double)(a) / (double)(b))
#define ExpectEq(a, b) ((a) == (b))
#define ExpectNe(a, b) ((a) != (b))
#define ExpectLe(a, b) ((a) <= (b))
#define ExpectLt(a, b) ((a) < (b))
#define LogicAnd(a, b) ((a) && (b))
#define LogicOr(a, b) ((a) || (b))
#define True true
#define False false
// AF_AXES_RESOURCE_END
