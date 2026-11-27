/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <limits>
#include <cfenv>
#include <cerrno>
#include <memory>
#include <initializer_list>
#pragma STDC FENV_ACCESS ON

template <typename A, typename B>
__attribute__((noinline)) auto Oracle(A a, B b) -> decltype(std::fmod(a, b)) {
  return std::fmod(a, b);
}
template <typename A, typename B>
__attribute__((noinline)) auto Actual(A a, B b) -> decltype(std::fmod(a, b)) {
  return Mod(a, b);
}
static uint64_t state = 0x91dfbade7531ULL;
static uint64_t checks = 0;
uint64_t Next() {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}
template <typename A, typename B>
bool Check(A a, B b) {
  std::feclearexcept(FE_ALL_EXCEPT);
  errno = 0;
  const auto expected = Oracle(a, b);
  const int expected_errno = errno;
  const int expected_flags = std::fetestexcept(FE_ALL_EXCEPT);
  std::feclearexcept(FE_ALL_EXCEPT);
  errno = 0;
  const auto actual = Actual(a, b);
  const int actual_errno = errno;
  const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT);
  ++checks;
  const bool equal =
      std::isnan(expected) ? std::isnan(actual) : expected == actual && std::signbit(expected) == std::signbit(actual);
  if (!equal || expected_errno != actual_errno || expected_flags != actual_flags) {
    std::printf("mismatch a=%La b=%La expected=%La actual=%La errno=%d/%d flags=%d/%d size=%zu/%zu\n", (long double)a,
                (long double)b, (long double)expected, (long double)actual, expected_errno, actual_errno,
                expected_flags, actual_flags, sizeof(A), sizeof(B));
    return false;
  }
  return true;
}
template <typename Float, typename Bits>
Float FloatFromBits(const Bits &bits) {
  static_assert(sizeof(Float) == sizeof(Bits), "The bit pattern must match the floating-point width");
  Float result;
  const auto *src = reinterpret_cast<const unsigned char *>(std::addressof(bits));
  auto *dst = reinterpret_cast<unsigned char *>(std::addressof(result));
  std::copy_n(src, sizeof(Float), dst);
  return result;
}

int main() {
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    if (std::fesetround(rounding) != 0) {
      return 2;
    }
    for (int n = 0; n < 20000; ++n) {
      const auto a = FloatFromBits<double>(Next());
      const auto b = FloatFromBits<double>(Next());
      const auto x = FloatFromBits<float>(static_cast<uint32_t>(Next()));
      const auto y = FloatFromBits<float>(static_cast<uint32_t>(Next()));
      if (!Check(a, b) || !Check(x, y)) {
        return 3;
      }
      if (!Check(static_cast<double>(uint32_t(Next())), static_cast<double>(uint32_t(Next()))) ||
          !Check(static_cast<double>(uint32_t(Next())) / 4.0, static_cast<double>(1 + uint32_t(Next() % 1024))) ||
          !Check(static_cast<float>(uint32_t(Next())), static_cast<float>(uint32_t(Next()))) ||
          !Check(static_cast<double>(uint32_t(Next())), int64_t(Next()))) {
        return 4;
      }
    }
    const double values[] = {0.0,
                             -0.0,
                             1.0,
                             -1.0,
                             0.5,
                             -0.5,
                             4294967295.0,
                             4294967296.0,
                             4294967297.0,
                             9007199254740992.0,
                             std::numeric_limits<double>::min(),
                             std::numeric_limits<double>::denorm_min(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()};
    for (auto a : values) {
      for (auto b : values) {
        if (!Check(a, b) || !Check((long double)a, (long double)b) || !Check((float)a, b) || !Check(a, (float)b)) {
          return 5;
        }
      }
    }
  }
  std::printf("verified=%llu rounding_modes=4 value+signed_zero+errno+fenv matched\n",
              static_cast<unsigned long long>(checks));
  return 0;
}
