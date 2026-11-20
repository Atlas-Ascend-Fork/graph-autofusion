/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

#include "autofuse_tiling_data.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void rand_seed_store_test(GM_ADDR x1, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

#if defined(RAND_E2E_DTYPE_U32)
using OutputDtype = uint32_t;
#elif defined(RAND_E2E_DTYPE_I32)
using OutputDtype = int32_t;
#elif defined(RAND_E2E_DTYPE_FP16)
using OutputDtype = half;
#elif defined(RAND_E2E_DTYPE_BF16)
using OutputDtype = bfloat16_t;
#else
using OutputDtype = float;
#endif

namespace {
constexpr uint32_t kMantissaMask = 0x7FFFFFU;
constexpr uint32_t kExponentMask = 127U << 23U;
constexpr uint32_t kBfloat16RoundBias = 0x7FFFU;

void PhiloxRound(uint32_t *counter, uint32_t *key) {
  const uint64_t res0 = static_cast<uint64_t>(0xD2511F53U) * counter[0];
  const uint64_t res1 = static_cast<uint64_t>(0xCD9E8D57U) * counter[2];
  counter[0] = static_cast<uint32_t>(res1 >> 32U) ^ counter[1] ^ key[0];
  counter[1] = static_cast<uint32_t>(res1);
  counter[2] = static_cast<uint32_t>(res0 >> 32U) ^ counter[3] ^ key[1];
  counter[3] = static_cast<uint32_t>(res0);
}

uint32_t GoldenPhiloxWord(int64_t seed, uint32_t offset) {
  uint32_t key[2] = {static_cast<uint32_t>(static_cast<uint64_t>(seed)),
                     static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32U)};
  uint32_t counter[4] = {offset, 0U, 0U, 0U};
  for (uint32_t i = 0U; i < 10U; ++i) {
    PhiloxRound(counter, key);
    key[0] += 0x9E3779B9U;
    key[1] += 0xBB67AE85U;
  }
  return counter[0];
}

float GoldenRand(int64_t seed, uint32_t offset) {
  const uint32_t word = GoldenPhiloxWord(seed, offset);
  union FloatBits {
    uint32_t bits;
    float value;
  };
  const FloatBits float_bits = {(word & kMantissaMask) | kExponentMask};
  return float_bits.value - 1.0F;
}

template <typename T>
float ConvertFloatToDtype(const float value) {
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    union FloatBits {
      uint32_t bits;
      float value;
    };
    FloatBits input = {};
    input.value = value;
    const uint32_t rounded = input.bits + kBfloat16RoundBias + ((input.bits >> 16U) & 1U);
    FloatBits output = {rounded & 0xFFFF0000U};
    return output.value;
  } else {
    return static_cast<float>(static_cast<T>(value));
  }
}

template <typename T>
float GoldenRandConverted(const int64_t seed, const uint32_t offset) {
  float expect = ConvertFloatToDtype<T>(GoldenRand(seed, offset));
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    if (expect >= 1.0F) {
      expect = 0.99609375F;
    }
  } else if constexpr (std::is_same_v<T, half>) {
    if (expect >= 1.0F) {
      expect = 0.99951171875F;
    }
  }
  return expect;
}
}  // namespace

class E2EBackendRandSeedStoreCode : public testing::Test, public testing::WithParamInterface<int> {};

TEST_P(E2EBackendRandSeedStoreCode, CalculateCorrect) {
  const int test_size = GetParam();
  const int64_t seed = std::numeric_limits<int64_t>::min();
  uint32_t block_dim = 48;

  AutofuseTilingData tiling_data;
  int64_t *seed_buffer = static_cast<int64_t *>(AscendC::GmAlloc(test_size * sizeof(int64_t) + 32));
  auto *y = static_cast<OutputDtype *>(AscendC::GmAlloc(test_size * sizeof(OutputDtype) + 32));
  for (int i = 0; i < test_size; ++i) {
    seed_buffer[i] = seed;
  }

  uint64_t workspace_size = 0;
  AutofuseTiling(test_size, &tiling_data, &workspace_size, &block_dim, 48, 192 * 1024);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(rand_seed_store_test, tiling_data.block_dim, reinterpret_cast<uint8_t *>(seed_buffer),
              reinterpret_cast<uint8_t *>(y), nullptr, reinterpret_cast<uint8_t *>(&tiling_data));

  uint32_t mismatch_count = 0;
  for (int i = 0; i < test_size; ++i) {
    if constexpr (std::is_same_v<OutputDtype, uint32_t> || std::is_same_v<OutputDtype, int32_t>) {
      const uint32_t word = GoldenPhiloxWord(seed, static_cast<uint32_t>(i));
      mismatch_count += (static_cast<uint32_t>(y[i]) != word) ? 1U : 0U;
    } else {
      const float value = static_cast<float>(y[i]);
      const float expect = GoldenRandConverted<OutputDtype>(seed, static_cast<uint32_t>(i));
      mismatch_count += (value < 0.0F || value >= 1.0F || value != expect) ? 1U : 0U;
    }
  }
  EXPECT_EQ(mismatch_count, 0U);

  AscendC::GmFree(seed_buffer);
  AscendC::GmFree(y);
}

INSTANTIATE_TEST_SUITE_P(CalculateWithBoundarySize, E2EBackendRandSeedStoreCode,
                         ::testing::Values(1, 3, 15, 17, 33, 65, 32, 64, 128, 256));
