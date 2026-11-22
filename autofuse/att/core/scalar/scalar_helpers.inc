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
inline bool IsEqual(double a, double b)
{
    const double epsilon = 1e-8;
    double abs = (a > b) ? (a - b) : (b - a);
    return abs < epsilon;
}
template<typename T1, typename T2>
inline double TernaryOp(bool cond, T1 a, T2 b)
{
    return static_cast<double>(cond ? a : b);
}
template<typename T>
inline T Ceiling(T a)
{
    T value = static_cast<T>(static_cast<int64_t>(a));
    return (IsEqual(value, a)) ? value : (value + 1);
}
template<typename T>
inline T Floor(T a)
{
    return static_cast<T>(static_cast<int64_t>(a));
}
template<typename T1, typename T2>
inline auto Mod(T1 a, T2 b)->decltype(a % b)
{
    return a % b;
}
template<typename T1, typename T2>
inline auto Mod(T1 a, T2 b)->typename std::enable_if<std::is_floating_point<T1>::value || std::is_floating_point<T2>::value, decltype(std::fmod(a, b))>::type
{
    return std::fmod(a, b);
}
template<typename TI, typename TO>
inline TO &RefToRef(TI &ptr) {
  return *(reinterpret_cast<TO *>(reinterpret_cast<void *>(&ptr)));
}

// AF_AXES_RESOURCE_END
