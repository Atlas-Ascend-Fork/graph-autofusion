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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/argmin.h"

using namespace AscendC;

namespace af {
namespace {
enum class MinPattern { kFinite, kTie, kNaN, kZero, kLimits, kAllMax, kInfinity, kAllNaN };
constexpr uint32_t kGuardBytes = 32U;

uint64_t MinAlignBytes(uint64_t bytes) {
  return (bytes + 31U) & ~31ULL;
}
void FreeMinGm(void *ptr) {
  if (ptr != nullptr) {
    GmFree(ptr);
  }
}
template <typename T>
using MinGm = std::unique_ptr<T, decltype(&FreeMinGm)>;
template <typename T>
MinGm<T> AllocateMinGm(uint64_t elements) {
  return MinGm<T>(static_cast<T *>(GmAlloc(MinAlignBytes(elements * sizeof(T)))), &FreeMinGm);
}

template <typename T>
bool MinIsNaN(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return std::isnan(value);
  }
  return false;
}

// Independent scalar definition: NaN first, then least value, then least supplied global index.
template <typename T>
bool PreferMin(T candidate, int32_t candidate_index, T best, int32_t best_index) {
  if (MinIsNaN(best)) {
    return MinIsNaN(candidate) && candidate_index < best_index;
  }
  if (MinIsNaN(candidate)) {
    return true;
  }
  return candidate < best || (candidate == best && candidate_index < best_index);
}

bool MinFloatSpecialValue(uint32_t a, uint32_t r, uint32_t extent, MinPattern pattern, float &value) {
  if (pattern == MinPattern::kAllNaN || (pattern == MinPattern::kNaN && (r == 1U || r + 1U == extent))) {
    const uint32_t bits = 0x7FC00000U | (r + 1U);
    std::copy_n(reinterpret_cast<const uint8_t *>(&bits), sizeof(value), reinterpret_cast<uint8_t *>(&value));
    return true;
  }
  if (pattern == MinPattern::kZero) {
    value = r == 0U ? (a % 2U == 0U ? 0.0F : -0.0F) : (a % 2U == 0U ? -0.0F : 0.0F);
    return true;
  }
  if (pattern == MinPattern::kAllMax) {
    value = std::numeric_limits<float>::infinity();
    return true;
  }
  if (pattern == MinPattern::kInfinity) {
    value = r % 3U == 1U ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    return true;
  }
  return false;
}

template <typename T>
T MinInputValue(uint32_t a, uint32_t r, uint32_t extent, MinPattern pattern) {
  if constexpr (std::is_same_v<T, float>) {
    float value;
    if (MinFloatSpecialValue(a, r, extent, pattern, value)) {
      return value;
    }
  }
  if (pattern == MinPattern::kAllMax) {
    return std::numeric_limits<T>::max();
  }
  if (pattern == MinPattern::kLimits) {
    return r == 1U || r + 1U == extent ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
  }
  if (pattern == MinPattern::kZero) {
    return static_cast<T>(0);
  }
  if (pattern == MinPattern::kTie && (r == 1U || r + 1U == extent)) {
    return static_cast<T>(-1000);
  }
  return static_cast<T>(static_cast<int32_t>((r * 17U + a * 13U) % 97U) - 48);
}

template <typename T>
struct MinSingleCase {
  uint32_t a, r, extent, columns, base;
  MinGm<T> input, values;
  MinGm<int32_t> indices;
  uint64_t value_count, index_count;

  MinSingleCase(uint32_t a_size, uint32_t r_size, uint32_t valid_r, uint32_t valid_a, uint32_t index_base)
      : a(a_size),
        r(r_size),
        extent(valid_r == 0U || valid_r > r ? r : valid_r),
        columns(valid_a == 0U || valid_a > a ? a : valid_a),
        base(index_base),
        input(AllocateMinGm<T>(MinAlignBytes(static_cast<uint64_t>(a) * r * sizeof(T)) / sizeof(T))),
        values(AllocateMinGm<T>(a + 2U * kGuardBytes / sizeof(T))),
        indices(AllocateMinGm<int32_t>(a + 2U * kGuardBytes / sizeof(int32_t))),
        value_count(MinAlignBytes(a * sizeof(T) + 2U * kGuardBytes) / sizeof(T)),
        index_count(MinAlignBytes(a * sizeof(int32_t) + 2U * kGuardBytes) / sizeof(int32_t)) {}

  template <bool ra>
  uint64_t Offset(uint32_t column, uint32_t row) const {
    return ra ? static_cast<uint64_t>(row) * a + column : static_cast<uint64_t>(column) * r + row;
  }

  template <bool ra>
  void Fill(MinPattern pattern) {
    ASSERT_TRUE(input && values && indices);
    std::fill_n(input.get(), MinAlignBytes(static_cast<uint64_t>(a) * r * sizeof(T)) / sizeof(T),
                std::numeric_limits<T>::lowest());
    for (uint32_t column = 0; column < a; ++column) {
      for (uint32_t row = 0; row < extent; ++row) {
        input.get()[Offset<ra>(column, row)] = MinInputValue<T>(column, row, extent, pattern);
      }
    }
    std::fill_n(values.get(), value_count, static_cast<T>(7777));
    std::fill_n(indices.get(), index_count, -7777);
  }

  template <bool ra, bool withValue>
  void Execute(uint32_t valid_r, uint32_t valid_a) {
    auto *src = input.get();
    auto *val = values.get();
    auto *idx = indices.get();
    const uint32_t aa = a, rr = r, index_base = base;
    auto kernel = [=] {
      TPipe pipe;
      TBuf<TPosition::VECCALC> src_buf, value_buf, index_buf;
      pipe.InitBuffer(src_buf, MinAlignBytes(static_cast<uint64_t>(aa) * rr * sizeof(T)));
      pipe.InitBuffer(value_buf, MinAlignBytes(aa * sizeof(T) + 2U * kGuardBytes));
      pipe.InitBuffer(index_buf, MinAlignBytes(aa * sizeof(int32_t) + 2U * kGuardBytes));
      auto src_local = src_buf.Get<T>();
      auto value_local = value_buf.Get<T>();
      auto index_local = index_buf.Get<int32_t>();
      GmToUb(src_local, src, aa * rr);
      GmToUb(value_local, val, aa + 2U * kGuardBytes / sizeof(T));
      GmToUb(index_local, idx, aa + 2U * kGuardBytes / sizeof(int32_t));
      const uint32_t shape[] = {ra ? rr : aa, ra ? aa : rr};
      if constexpr (withValue) {
        ArgMinWithValueExtend<T, std::conditional_t<ra, Pattern::Reduce::RA, Pattern::Reduce::AR>>(
            index_local[kGuardBytes / sizeof(int32_t)], value_local[kGuardBytes / sizeof(T)], src_local, shape, false,
            index_base, valid_r, valid_a);
      } else {
        ArgMinExtend<T, std::conditional_t<ra, Pattern::Reduce::RA, Pattern::Reduce::AR>>(
            index_local[kGuardBytes / sizeof(int32_t)], src_local, shape, false, valid_r, valid_a);
      }
      UbToGm(val, value_local, aa + 2U * kGuardBytes / sizeof(T));
      UbToGm(idx, index_local, aa + 2U * kGuardBytes / sizeof(int32_t));
    };
    SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);
  }

  template <bool ra, bool withValue>
  void Verify() const {
    for (uint32_t column = 0; column < columns; ++column) {
      T best = input.get()[Offset<ra>(column, 0U)];
      int32_t winner = static_cast<int32_t>(base);
      for (uint32_t row = 1U; row < extent; ++row) {
        const T next = input.get()[Offset<ra>(column, row)];
        const int32_t next_index = static_cast<int32_t>(base + row);
        if (PreferMin(next, next_index, best, winner)) {
          best = next;
          winner = next_index;
        }
      }
      EXPECT_EQ(indices.get()[kGuardBytes / sizeof(int32_t) + column], winner);
      if constexpr (withValue) {
        EXPECT_EQ(std::memcmp(values.get() + kGuardBytes / sizeof(T) + column, &best, sizeof(T)), 0);
      }
    }
    for (uint64_t i = 0; i < index_count; ++i) {
      if (i < kGuardBytes / sizeof(int32_t) || i >= kGuardBytes / sizeof(int32_t) + columns) {
        EXPECT_EQ(indices.get()[i], -7777) << i;
      }
    }
    for (uint64_t i = 0; i < value_count; ++i) {
      if (!withValue || i < kGuardBytes / sizeof(T) || i >= kGuardBytes / sizeof(T) + columns) {
        EXPECT_EQ(values.get()[i], static_cast<T>(7777)) << i;
      }
    }
  }
};

template <typename T, bool ra, bool withValue>
void RunMinSingleSuite() {
  const std::array<uint32_t, 4> shapes[] = {{1, 1, 0, 0},    {3, 32, 31, 0},  {3, 65, 65, 0}, {4, 129, 97, 1},
                                            {31, 65, 64, 0}, {32, 33, 32, 0}, {33, 65, 0, 0}, {64, 7, 7, 0},
                                            {65, 3, 3, 0},   {128, 7, 6, 0},  {256, 3, 3, 0}};
  for (const auto &shape : shapes) {
    for (const auto pattern : {MinPattern::kFinite, MinPattern::kTie, MinPattern::kNaN, MinPattern::kZero,
                               MinPattern::kLimits, MinPattern::kAllMax, MinPattern::kInfinity, MinPattern::kAllNaN}) {
      SCOPED_TRACE(shape[0]);
      SCOPED_TRACE(shape[1]);
      SCOPED_TRACE(static_cast<int>(pattern));
      MinSingleCase<T> data(shape[0], shape[1], shape[2], shape[3], withValue ? 37U : 0U);
      ASSERT_NO_FATAL_FAILURE((data.template Fill<ra>(pattern)));
      data.template Execute<ra, withValue>(shape[2], shape[3]);
      ASSERT_NO_FATAL_FAILURE((data.template Verify<ra, withValue>()));
    }
  }
}

TEST(TestApiArgmin, IndexFloat32LayoutsTailsNaNsAndSignedZero) {
  RunMinSingleSuite<float, false, false>();
  RunMinSingleSuite<float, true, false>();
}
TEST(TestApiArgmin, IndexInt32LayoutsAndLimits) {
  RunMinSingleSuite<int32_t, false, false>();
  RunMinSingleSuite<int32_t, true, false>();
}
TEST(TestApiArgmin, WithValueFloat32PreservesWinnerBits) {
  RunMinSingleSuite<float, false, true>();
  RunMinSingleSuite<float, true, true>();
}
TEST(TestApiArgmin, WithValueInt32PreservesValuesAndGlobalIndices) {
  RunMinSingleSuite<int32_t, false, true>();
  RunMinSingleSuite<int32_t, true, true>();
}
TEST(TestApiArgmin, WithValueInt64HandlesMinimumWithoutNegation) {
  RunMinSingleSuite<int64_t, false, true>();
  RunMinSingleSuite<int64_t, true, true>();
}
}  // namespace
}  // namespace af
