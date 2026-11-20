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
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <gtest/gtest.h>
#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

#ifdef ARGMIN_WITH_VALUE
extern "C" __global__ __aicore__ void argmin_value_store_test(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
#else
extern "C" __global__ __aicore__ void argmin_store_test(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
#endif
extern "C" int64_t AutofuseTiling(uint32_t, uint32_t, AutofuseTilingData *, uint64_t *, uint32_t *, uint32_t, uint32_t);

namespace {
#if defined(ARGMIN_I64)
using Value = int64_t;
#elif defined(ARGMIN_I32)
using Value = int32_t;
#else
using Value = float;
#endif
#ifdef ARGMIN_RA
constexpr bool kRa = true;
#else
constexpr bool kRa = false;
#endif
void FreeGm(void *p) {
  if (p != nullptr) {
    AscendC::GmFree(p);
  }
}
using GmOwner = std::unique_ptr<void, decltype(&FreeGm)>;

Value InputValue(uint32_t a, uint32_t r, uint32_t rows, bool special) {
  if constexpr (std::is_same_v<Value, float>) {
    if (special && a % 3U == 0U) {
      return r == 0U ? 0.0F : -0.0F;
    }
    if (special && a % 3U == 1U && (r == 1U || r + 1U == rows)) {
      const uint32_t bits = 0x7FC00000U | (r + 1U);
      Value value;
      std::copy_n(reinterpret_cast<const uint8_t *>(&bits), sizeof(value), reinterpret_cast<uint8_t *>(&value));
      return value;
    }
    if (special && a % 3U == 2U) {
      return r % 3U == 0U ? std::numeric_limits<Value>::infinity() : -std::numeric_limits<Value>::infinity();
    }
  }
  if (special && !std::is_same_v<Value, float>) {
    if (a % 2U == 0U) {
      return r + 1U == rows ? std::numeric_limits<Value>::max() - 1 : std::numeric_limits<Value>::max();
    }
    return r + 1U == rows ? std::numeric_limits<Value>::lowest() : std::numeric_limits<Value>::lowest() + 1;
  }
  if (r == 1U || r + 1U == rows) {
    return std::numeric_limits<Value>::lowest();
  }
  return static_cast<Value>((static_cast<int64_t>(a) * 7 + r * 13) % 997 - 498);
}

bool IsNan(Value x) {
  if constexpr (std::is_same_v<Value, float>) {
    return std::isnan(x);
  }
  return false;
}

struct MinE2ECase {
  uint32_t m, n, a, r, blocks{0};
  uint64_t workspace_bytes{0};
  AutofuseTilingData tiling{};
  GmOwner input{nullptr, &FreeGm}, values{nullptr, &FreeGm}, indices{nullptr, &FreeGm}, workspace{nullptr, &FreeGm};

  MinE2ECase(uint32_t rows, uint32_t columns) : m(rows), n(columns), a(kRa ? n : m), r(kRa ? m : n) {}

  uint64_t Offset(uint32_t column, uint32_t row) const {
    return kRa ? static_cast<uint64_t>(row) * n + column : static_cast<uint64_t>(column) * n + row;
  }

  void Prepare(bool special) {
    ASSERT_EQ(AutofuseTiling(m, n, &tiling, &workspace_bytes, &blocks, 48, 192U * 1024U), 0);
    ASSERT_GT(blocks, 0U);
    input.reset(AscendC::GmAlloc(static_cast<uint64_t>(m) * n * sizeof(Value)));
    values.reset(AscendC::GmAlloc(a * sizeof(Value)));
    indices.reset(AscendC::GmAlloc(a * sizeof(int64_t)));
    workspace.reset(AscendC::GmAlloc(std::max<uint64_t>(workspace_bytes, 32U)));
    ASSERT_TRUE(input && values && indices && workspace);
    std::fill_n(static_cast<uint8_t *>(workspace.get()), std::max<uint64_t>(workspace_bytes, 32U), 0xFFU);
    std::fill_n(static_cast<int64_t *>(indices.get()), a, -1);
    std::fill_n(static_cast<Value *>(values.get()), a, static_cast<Value>(7777));
    for (uint32_t column = 0; column < a; ++column) {
      for (uint32_t row = 0; row < r; ++row) {
        static_cast<Value *>(input.get())[Offset(column, row)] = InputValue(column, row, r, special);
      }
    }
  }

  void Execute() {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
#ifdef ARGMIN_WITH_VALUE
    ICPU_RUN_KF(argmin_value_store_test, blocks, static_cast<uint8_t *>(input.get()),
                static_cast<uint8_t *>(values.get()), static_cast<uint8_t *>(indices.get()),
                static_cast<uint8_t *>(workspace.get()), reinterpret_cast<uint8_t *>(&tiling));
#else
    ICPU_RUN_KF(argmin_store_test, blocks, static_cast<uint8_t *>(input.get()), static_cast<uint8_t *>(indices.get()),
                static_cast<uint8_t *>(workspace.get()), reinterpret_cast<uint8_t *>(&tiling));
#endif
  }

  void Verify() const {
    for (uint32_t column = 0; column < a; ++column) {
      Value best = static_cast<Value *>(input.get())[Offset(column, 0U)];
      int64_t winner = 0;
      for (uint32_t row = 1; row < r; ++row) {
        const Value candidate = static_cast<Value *>(input.get())[Offset(column, row)];
        if (!IsNan(best) && (IsNan(candidate) || candidate < best)) {
          best = candidate;
          winner = row;
        }
      }
      EXPECT_EQ(static_cast<int64_t *>(indices.get())[column], winner) << column;
#ifdef ARGMIN_WITH_VALUE
      EXPECT_EQ(std::memcmp(static_cast<Value *>(values.get()) + column, &best, sizeof(Value)), 0) << column;
#endif
    }
  }
};

void RunMinE2E(bool special) {
  for (const auto &shape : {std::pair<uint32_t, uint32_t>{3, 65}, {8, 4096}, {1, 129}}) {
    SCOPED_TRACE(shape.first);
    SCOPED_TRACE(shape.second);
    SCOPED_TRACE(special);
    MinE2ECase data(shape.first, shape.second);
    ASSERT_NO_FATAL_FAILURE(data.Prepare(special));
    data.Execute();
    ASSERT_NO_FATAL_FAILURE(data.Verify());
  }
}
TEST(BackendArgminKernel, FiniteTiesAndIntegerMinimum) {
  RunMinE2E(false);
}
#if defined(ARGMIN_I32) || defined(ARGMIN_I64)
TEST(BackendArgminKernel, AdjacentIntegerExtremesKeepFullPrecision) {
  RunMinE2E(true);
}
#else
TEST(BackendArgminKernel, NaNsInfinitiesAndSignedZero) {
  RunMinE2E(true);
}
#endif
}  // namespace
