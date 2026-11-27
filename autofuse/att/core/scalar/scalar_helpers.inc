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
    // Classify IEEE binary32/binary64 bits before converting nonnegative integers.
    // Negative zero, zero divisors, fractions and exceptional values retain the fmod path.
    using Result = decltype(std::fmod(a, b));
    const auto as_uint32 = [](Result value, uint32_t &integer) {
      if (!((std::is_same<Result, float>::value && sizeof(Result) == 4U) ||
            (std::is_same<Result, double>::value && sizeof(Result) == 8U))) {
        return false;
      }
      using Bits = typename std::conditional<sizeof(Result) == 8U, uint64_t, uint32_t>::type;
      constexpr uint32_t kFractionBits = sizeof(Result) == 8U ? 52U : 23U;
      constexpr uint32_t kExponentBias = sizeof(Result) == 8U ? 1023U : 127U;
      Bits bits = 0;
      __builtin_memcpy(&bits, &value, sizeof(bits));
      if (bits == 0U) {
        integer = 0U;
        return true;
      }
      const auto exponent_with_sign = bits >> kFractionBits;
      if (exponent_with_sign < kExponentBias || exponent_with_sign >= kExponentBias + 32U) {
        return false;
      }
      const auto exponent = exponent_with_sign - kExponentBias;
      const Bits fraction_mask =
          exponent >= kFractionBits ? Bits{0} : (Bits{1} << (kFractionBits - exponent)) - 1U;
      if ((bits & fraction_mask) != 0U) {
        return false;
      }
      integer = static_cast<uint32_t>(value);
      return true;
    };
    uint32_t left = 0U;
    uint32_t right = 0U;
    if (as_uint32(static_cast<Result>(a), left) && as_uint32(static_cast<Result>(b), right) && right != 0U) {
      return static_cast<Result>(left % right);
    }
    return std::fmod(a, b);
}
template<typename TI, typename TO>
inline TO &RefToRef(TI &ptr) {
  return *(reinterpret_cast<TO *>(reinterpret_cast<void *>(&ptr)));
}

// AF_AXES_RESOURCE_END
