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
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "test_api_utils.h"
#include "tikicpulib.h"
#include "api_regbase/random.h"

using namespace AscendC;

namespace {
constexpr uint32_t kMantissaMask = 0x7FFFFFU;
constexpr uint32_t kExponentMask = 127U << 23U;
constexpr float kMinUniform = 1e-7F;
constexpr float kTwoPi = 6.283185307179586F;

void PhiloxRound(uint32_t *counter, uint32_t *key) {
  const uint64_t res0 = static_cast<uint64_t>(0xD2511F53U) * counter[0];
  const uint64_t res1 = static_cast<uint64_t>(0xCD9E8D57U) * counter[2];
  const uint32_t res0Lo = static_cast<uint32_t>(res0);
  const uint32_t res0Hi = static_cast<uint32_t>(res0 >> 32U);
  const uint32_t res1Lo = static_cast<uint32_t>(res1);
  const uint32_t res1Hi = static_cast<uint32_t>(res1 >> 32U);
  counter[0] = res1Hi ^ counter[1] ^ key[0];
  counter[1] = res1Lo;
  counter[2] = res0Hi ^ counter[3] ^ key[1];
  counter[3] = res0Lo;
}

uint32_t GoldenPhiloxWord(int64_t seed, uint32_t offset, uint32_t word) {
  uint32_t key[2] = {static_cast<uint32_t>(static_cast<uint64_t>(seed) & 0xFFFFFFFFU),
                     static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32U)};
  uint32_t counter[4] = {offset, 0U, 0U, 0U};
  for (uint32_t i = 0; i < 10U; ++i) {
    PhiloxRound(counter, key);
    key[0] += 0x9E3779B9U;
    key[1] += 0xBB67AE85U;
  }
  return counter[word];
}

template <typename T>
void CheckRandnStatistics(const std::vector<T> &result) {
  double sum = 0.0;
  for (const auto value : result) {
    sum += static_cast<double>(value);
  }
  const double mean = sum / static_cast<double>(result.size());
  double square_sum = 0.0;
  for (const auto value : result) {
    const double diff = static_cast<double>(value) - mean;
    square_sum += diff * diff;
  }
  const double variance = square_sum / static_cast<double>(result.size());
  EXPECT_NEAR(mean, 0.0, 0.2);
  EXPECT_NEAR(variance, 1.0, 0.2);
}

float GoldenRand(int64_t seed, uint32_t offset) {
  union FloatBits {
    uint32_t bits;
    float value;
  };
  FloatBits floatBits = {(GoldenPhiloxWord(seed, offset, 0U) & kMantissaMask) | kExponentMask};
  return floatBits.value - 1.0F;
}

float GoldenRandFromWord(const uint32_t word) {
  union FloatBits {
    uint32_t bits;
    float value;
  };
  FloatBits floatBits = {(word & kMantissaMask) | kExponentMask};
  return floatBits.value - 1.0F;
}

float GoldenRandn(int64_t seed, uint32_t offset) {
  const float uniformX = std::max(GoldenRand(seed, offset), kMinUniform);
  const float uniformY = GoldenRandFromWord(GoldenPhiloxWord(seed, offset, 1U));
  return std::sqrt(-2.0F * std::log(uniformX)) * std::cos(kTwoPi * uniformY);
}

float GoldenVectorRand(const uint32_t element_index) {
  const uint32_t block_index = element_index / 4U;
  const uint32_t word_index = element_index % 4U;
  return GoldenRandFromWord(GoldenPhiloxWord(0, block_index, word_index));
}

template <typename T>
std::vector<T> RunRandnSeedOffset(int64_t seed, const std::vector<int32_t> &offset) {
  T *allocatedResult = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * offset.size() + 32U));
  const uintptr_t alignedAddress = (reinterpret_cast<uintptr_t>(allocatedResult) + 31U) & ~static_cast<uintptr_t>(31U);
  T *resultPtr = reinterpret_cast<T *>(alignedAddress);
  const uint32_t size = static_cast<uint32_t>(offset.size());
  const uint32_t aligned_size = (size + 15U) / 16U * 16U;
  auto kernel = [&] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> dstBuf;
    TBuf<TPosition::VECCALC> seedBuf;
    TBuf<TPosition::VECCALC> offsetBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    tpipe.InitBuffer(dstBuf, sizeof(T) * size);
    tpipe.InitBuffer(seedBuf, sizeof(int64_t) * aligned_size);
    tpipe.InitBuffer(offsetBuf, sizeof(int32_t) * size);
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);
    LocalTensor<T> dst = dstBuf.Get<T>();
    LocalTensor<int64_t> seedTensor = seedBuf.Get<int64_t>();
    LocalTensor<int32_t> offsetTensor = offsetBuf.Get<int32_t>();
    LocalTensor<uint8_t> tmp = tmpBuf.Get<uint8_t>();
    for (uint32_t i = 0U; i < size; ++i) {
      seedTensor.SetValue(i, seed);
      offsetTensor.SetValue(i, offset[i]);
    }
    RandnSeedOffset(dst, seedTensor, offsetTensor, tmp, size);
    UbToGm(resultPtr, dst, size);
  };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  std::vector<T> result(resultPtr, resultPtr + size);
  AscendC::GmFree(allocatedResult);
  return result;
}

template <typename T>
std::vector<T> RunRandSeedOffset(int64_t seed, const std::vector<int32_t> &offset) {
  T *allocatedResult = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * offset.size() + 32U));
  const uintptr_t alignedAddress = (reinterpret_cast<uintptr_t>(allocatedResult) + 31U) & ~static_cast<uintptr_t>(31U);
  T *resultPtr = reinterpret_cast<T *>(alignedAddress);
  const uint32_t size = static_cast<uint32_t>(offset.size());
  auto kernel = [&] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> dstBuf;
    TBuf<TPosition::VECCALC> offsetBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    tpipe.InitBuffer(dstBuf, sizeof(T) * size);
    tpipe.InitBuffer(offsetBuf, sizeof(int32_t) * size);
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);
    LocalTensor<T> dst = dstBuf.Get<T>();
    LocalTensor<int32_t> offsetTensor = offsetBuf.Get<int32_t>();
    LocalTensor<uint8_t> tmp = tmpBuf.Get<uint8_t>();
    for (uint32_t i = 0U; i < size; ++i) {
      offsetTensor.SetValue(i, offset[i]);
    }
    RandSeedOffset(dst, seed, offsetTensor, tmp, size);
    UbToGm(resultPtr, dst, size);
  };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  std::vector<T> result(resultPtr, resultPtr + size);
  AscendC::GmFree(allocatedResult);
  return result;
}

template <typename T>
std::vector<T> RunRandnScalarSeedOffset(int64_t seed, const std::vector<int32_t> &offset) {
  T *allocatedResult = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * offset.size() + 32U));
  const uintptr_t alignedAddress = (reinterpret_cast<uintptr_t>(allocatedResult) + 31U) & ~static_cast<uintptr_t>(31U);
  T *resultPtr = reinterpret_cast<T *>(alignedAddress);
  const uint32_t size = static_cast<uint32_t>(offset.size());
  auto kernel = [&] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> dstBuf;
    TBuf<TPosition::VECCALC> offsetBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    tpipe.InitBuffer(dstBuf, sizeof(T) * size);
    tpipe.InitBuffer(offsetBuf, sizeof(int32_t) * size);
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);
    LocalTensor<T> dst = dstBuf.Get<T>();
    LocalTensor<int32_t> offsetTensor = offsetBuf.Get<int32_t>();
    LocalTensor<uint8_t> tmp = tmpBuf.Get<uint8_t>();
    for (uint32_t i = 0U; i < size; ++i) {
      offsetTensor.SetValue(i, offset[i]);
    }
    RandnSeedOffset(dst, seed, offsetTensor, tmp, size);
    UbToGm(resultPtr, dst, size);
  };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  std::vector<T> result(resultPtr, resultPtr + size);
  AscendC::GmFree(allocatedResult);
  return result;
}

float GoldenBfloat16(const float value) {
  union FloatBits {
    uint32_t bits;
    float value;
  };
  constexpr uint32_t kBfloat16RoundBias = 0x7FFFU;
  FloatBits input = {};
  input.value = value;
  const uint32_t rounded = input.bits + kBfloat16RoundBias + ((input.bits >> 16U) & 1U);
  FloatBits output = {rounded & 0xFFFF0000U};
  return output.value;
}

template <typename T>
std::vector<T> RunRandLegacy(const uint32_t size) {
  T *allocatedResult = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * size + 32U));
  const uintptr_t alignedAddress = (reinterpret_cast<uintptr_t>(allocatedResult) + 31U) & ~static_cast<uintptr_t>(31U);
  T *resultPtr = reinterpret_cast<T *>(alignedAddress);
  auto kernel = [size, resultPtr] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> dstBuf;
    tpipe.InitBuffer(dstBuf, sizeof(T) * size);
    LocalTensor<T> dst = dstBuf.Get<T>();
    Rand(dst, size);
    UbToGm(resultPtr, dst, size);
  };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  std::vector<T> result(resultPtr, resultPtr + size);
  AscendC::GmFree(allocatedResult);
  return result;
}

template <typename T>
std::vector<T> RunRandnLegacy(const uint32_t size) {
  T *allocatedResult = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * size + 32U));
  const uintptr_t alignedAddress = (reinterpret_cast<uintptr_t>(allocatedResult) + 31U) & ~static_cast<uintptr_t>(31U);
  T *resultPtr = reinterpret_cast<T *>(alignedAddress);
  auto kernel = [size, resultPtr] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> dstBuf;
    tpipe.InitBuffer(dstBuf, sizeof(T) * size);
    LocalTensor<T> dst = dstBuf.Get<T>();
    Randn(dst, size);
    UbToGm(resultPtr, dst, size);
  };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  std::vector<T> result(resultPtr, resultPtr + size);
  AscendC::GmFree(allocatedResult);
  return result;
}

template <typename T>
float GoldenRandConverted(const uint32_t offset) {
  const float value = GoldenRand(0, offset);
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    const float converted = GoldenBfloat16(value);
    return converted >= 1.0F ? 0.99609375F : converted;
  } else {
    const float converted = static_cast<float>(static_cast<T>(value));
    return converted >= 1.0F ? 0.99951171875F : converted;
  }
}

template <typename T>
float GoldenRandnConverted(const uint32_t offset) {
  const float value = GoldenRandn(0, offset);
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    return GoldenBfloat16(value);
  } else {
    return static_cast<float>(static_cast<T>(value));
  }
}
}  // namespace

TEST(TestApiRandSeedOffset, RandMatchesHostGolden) {
  const int64_t seed = 42;
  const std::vector<int32_t> offset = {0, 1, 2, 3, 17, 128, 4096, 129800};
  const uint32_t size = static_cast<uint32_t>(offset.size());
  float *allocatedResult = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * size + 32U));
  const uintptr_t alignedAddress = (reinterpret_cast<uintptr_t>(allocatedResult) + 31U) & ~static_cast<uintptr_t>(31U);
  float *resultPtr = reinterpret_cast<float *>(alignedAddress);
  auto kernel = [&] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> dstBuf;
    TBuf<TPosition::VECCALC> offsetBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    tpipe.InitBuffer(dstBuf, sizeof(float) * size);
    tpipe.InitBuffer(offsetBuf, sizeof(int32_t) * size);
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);
    LocalTensor<float> dst = dstBuf.Get<float>();
    LocalTensor<int32_t> offsetTensor = offsetBuf.Get<int32_t>();
    LocalTensor<uint8_t> tmp = tmpBuf.Get<uint8_t>();
    for (uint32_t i = 0U; i < size; ++i) {
      offsetTensor.SetValue(i, offset[i]);
    }
    RandSeedOffset(dst, seed, offsetTensor, tmp, size);
    UbToGm(resultPtr, dst, size);
  };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  std::vector<float> result(resultPtr, resultPtr + size);
  AscendC::GmFree(allocatedResult);
  for (uint32_t i = 0U; i < size; ++i) {
    const float expect = GoldenRand(seed, static_cast<uint32_t>(offset[i]));
    EXPECT_FLOAT_EQ(result[i], expect);
  }
}

TEST(TestApiRandSeedOffset, RandOutputsPhiloxWord) {
  const int64_t seed = -123456789123456789LL;
  const std::vector<int32_t> offset = {0, 1, 2, 3, 7, 33};
  const auto result = RunRandSeedOffset<uint32_t>(seed, offset);
  for (size_t i = 0U; i < offset.size(); ++i) {
    EXPECT_EQ(result[i], GoldenPhiloxWord(seed, static_cast<uint32_t>(offset[i]), 0U));
  }
}

TEST(TestApiRandSeedOffset, RandExtremeSeedAndOffsetMatchHostGolden) {
  const std::vector<int64_t> seeds = {0, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
  const std::vector<int32_t> offsets = {0, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
  for (const int64_t seed : seeds) {
    const auto result = RunRandSeedOffset<float>(seed, offsets);
    for (size_t i = 0U; i < offsets.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], GoldenRand(seed, static_cast<uint32_t>(offsets[i])));
    }
  }
}

TEST(TestApiRandSeedOffset, RandInt32MatchesHostGolden) {
  const int64_t seed = -123456789123456789LL;
  const std::vector<int32_t> offset = {0, 1, 2, 3, 7, 33};
  const auto result = RunRandSeedOffset<int32_t>(seed, offset);
  for (size_t i = 0U; i < offset.size(); ++i) {
    EXPECT_EQ(result[i], static_cast<int32_t>(GoldenPhiloxWord(seed, static_cast<uint32_t>(offset[i]), 0U)));
  }
}

TEST(TestApiRandSeedOffset, RandHalfMatchesHostGolden) {
  const int64_t seed = 42;
  const std::vector<int32_t> offset = {0, 1, 2, 3, 17, 128};
  const auto result = RunRandSeedOffset<half>(seed, offset);
  for (size_t i = 0U; i < offset.size(); ++i) {
    float expect = static_cast<float>(static_cast<half>(GoldenRand(seed, static_cast<uint32_t>(offset[i]))));
    if (expect >= 1.0F) {
      expect = 0.99951171875F;
    }
    EXPECT_FLOAT_EQ(static_cast<float>(result[i]), expect);
  }
}

TEST(TestApiRandSeedOffset, RandBfloat16MatchesHostGolden) {
  const int64_t seed = 42;
  const std::vector<int32_t> offset = {0, 1, 2, 3, 17, 128};
  const auto result = RunRandSeedOffset<bfloat16_t>(seed, offset);
  for (size_t i = 0U; i < offset.size(); ++i) {
    float expect = GoldenBfloat16(GoldenRand(seed, static_cast<uint32_t>(offset[i])));
    if (expect >= 1.0F) {
      expect = 0.99609375F;
    }
    EXPECT_FLOAT_EQ(static_cast<float>(result[i]), expect);
  }
}

TEST(TestApiRandSeedOffset, RandLegacyHalfAndBfloat16MatchFixedKeyGolden) {
  constexpr uint32_t kSize = 65U;
  const auto halfResult = RunRandLegacy<half>(kSize);
  const auto bfloat16Result = RunRandLegacy<bfloat16_t>(kSize);
  for (uint32_t i = 0U; i < kSize; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(halfResult[i]), (GoldenRandConverted<half>(i)));
    EXPECT_FLOAT_EQ(static_cast<float>(bfloat16Result[i]), (GoldenRandConverted<bfloat16_t>(i)));
  }
}

TEST(TestApiRandSeedOffset, RandLegacyFloatMatchesVectorGolden) {
  constexpr uint32_t kSize = 16U;
  const auto result = RunRandLegacy<float>(kSize);
  for (uint32_t i = 0U; i < kSize; ++i) {
    EXPECT_FLOAT_EQ(result[i], GoldenVectorRand(i));
  }
}

TEST(TestApiRandSeedOffset, RandLegacyUint32AndInt32MatchFixedKeyGolden) {
  constexpr uint32_t kSize = 65U;
  const auto uint32Result = RunRandLegacy<uint32_t>(kSize);
  const auto int32Result = RunRandLegacy<int32_t>(kSize);
  for (uint32_t i = 0U; i < kSize; ++i) {
    const uint32_t expected = GoldenPhiloxWord(0, i / 4U, i % 4U);
    EXPECT_EQ(uint32Result[i], expected);
    EXPECT_EQ(static_cast<uint32_t>(int32Result[i]), expected);
  }
}

TEST(TestApiRandSeedOffset, RandnLegacyHalfAndBfloat16MatchFixedKeyGolden) {
  constexpr uint32_t kSize = 65U;
  const auto halfResult = RunRandnLegacy<half>(kSize);
  const auto bfloat16Result = RunRandnLegacy<bfloat16_t>(kSize);
  for (uint32_t i = 0U; i < kSize; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(halfResult[i]), (GoldenRandnConverted<half>(i)));
    EXPECT_FLOAT_EQ(static_cast<float>(bfloat16Result[i]), (GoldenRandnConverted<bfloat16_t>(i)));
  }
}

TEST(TestApiRandSeedOffset, RandnLegacyUint32AndInt32MatchFixedKeyGolden) {
  constexpr uint32_t kSize = 65U;
  const auto uint32Result = RunRandnLegacy<uint32_t>(kSize);
  const auto int32Result = RunRandnLegacy<int32_t>(kSize);
  for (uint32_t i = 0U; i < kSize; ++i) {
    const uint32_t expected = GoldenPhiloxWord(0, i / 4U, i % 4U);
    EXPECT_EQ(uint32Result[i], expected);
    EXPECT_EQ(static_cast<uint32_t>(int32Result[i]), expected);
  }
}

TEST(TestApiRandSeedOffset, LegacyFloatAndHalfKeepDistributionSemantics) {
  constexpr uint32_t kSize = 4096U;
  CheckRandnStatistics(RunRandnLegacy<float>(kSize));
  CheckRandnStatistics(RunRandnLegacy<half>(kSize));
  CheckRandnStatistics(RunRandnLegacy<bfloat16_t>(kSize));
}

TEST(TestApiRandSeedOffset, RandnOutputsPhiloxWord) {
  const int64_t seed = -123456789123456789LL;
  const std::vector<int32_t> offset = {0, 1, 2, 3, 7, 33};
  const auto result = RunRandnSeedOffset<uint32_t>(seed, offset);
  for (size_t i = 0U; i < offset.size(); ++i) {
    EXPECT_EQ(result[i], GoldenPhiloxWord(seed, static_cast<uint32_t>(offset[i]), 0U));
  }
}

TEST(TestApiRandSeedOffset, RandnExtremeSeedAndOffsetMatchHostGolden) {
  const std::vector<int64_t> seeds = {0, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
  const std::vector<int32_t> offsets = {0, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
  for (const int64_t seed : seeds) {
    const auto result = RunRandnSeedOffset<float>(seed, offsets);
    for (size_t i = 0U; i < offsets.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], GoldenRandn(seed, static_cast<uint32_t>(offsets[i])));
    }
  }
}

TEST(TestApiRandSeedOffset, RandBoundarySizesMatchHostGolden) {
  const int64_t seed = 987654321;
  const std::vector<int32_t> offsets = {0, 1, 2, 3, 15, 17, 31, 33, 63, 65};
  const auto result = RunRandSeedOffset<float>(seed, offsets);
  for (size_t i = 0U; i < offsets.size(); ++i) {
    EXPECT_FLOAT_EQ(result[i], GoldenRand(seed, static_cast<uint32_t>(offsets[i])));
  }
}

TEST(TestApiRandSeedOffset, RandnBoundarySizesMatchHostGolden) {
  const int64_t seed = 987654321;
  const std::vector<int32_t> offsets = {0, 1, 2, 3, 15, 17, 31, 33, 63, 65};
  const auto result = RunRandnSeedOffset<float>(seed, offsets);
  for (size_t i = 0U; i < offsets.size(); ++i) {
    EXPECT_FLOAT_EQ(result[i], GoldenRandn(seed, static_cast<uint32_t>(offsets[i])));
  }
}

TEST(TestApiRandSeedOffset, RandnInt32MatchesHostGolden) {
  const int64_t seed = -123456789123456789LL;
  const std::vector<int32_t> offset = {0, 1, 2, 3, 7, 33};
  const auto result = RunRandnSeedOffset<int32_t>(seed, offset);
  for (size_t i = 0U; i < offset.size(); ++i) {
    EXPECT_EQ(result[i], static_cast<int32_t>(GoldenPhiloxWord(seed, static_cast<uint32_t>(offset[i]), 0U)));
  }
}

TEST(TestApiRandSeedOffset, RandnFloatMatchesHostGolden) {
  const int64_t seed = -123456789123456789LL;
  const std::vector<int32_t> offset = {0, 1, 2, 3, 17, 128};
  const auto result = RunRandnSeedOffset<float>(seed, offset);
  for (size_t i = 0U; i < offset.size(); ++i) {
    EXPECT_FLOAT_EQ(result[i], GoldenRandn(seed, static_cast<uint32_t>(offset[i])));
  }
}

TEST(TestApiRandSeedOffset, RandnHalfMatchesHostGolden) {
  const int64_t seed = -123456789123456789LL;
  const std::vector<int32_t> offset = {0, 1, 2, 3, 17, 128};
  const auto result = RunRandnSeedOffset<half>(seed, offset);
  for (size_t i = 0U; i < offset.size(); ++i) {
    const float expect = static_cast<half>(GoldenRandn(seed, static_cast<uint32_t>(offset[i])));
    EXPECT_FLOAT_EQ(static_cast<float>(result[i]), expect);
  }
}

TEST(TestApiRandSeedOffset, RandnBfloat16MatchesHostGolden) {
  const int64_t seed = -123456789123456789LL;
  const std::vector<int32_t> offset = {0, 1, 2, 3, 17, 128};
  const auto result = RunRandnSeedOffset<bfloat16_t>(seed, offset);
  for (size_t i = 0U; i < offset.size(); ++i) {
    const float expect = GoldenBfloat16(GoldenRandn(seed, static_cast<uint32_t>(offset[i])));
    EXPECT_FLOAT_EQ(static_cast<float>(result[i]), expect);
  }
}

TEST(TestApiRandSeedOffset, RandnFloatHasNormalDistributionStatistics) {
  const int64_t seed = 20261117;
  std::vector<int32_t> offset(4096);
  for (size_t i = 0U; i < offset.size(); ++i) {
    offset[i] = static_cast<int32_t>(i);
  }
  CheckRandnStatistics(RunRandnSeedOffset<float>(seed, offset));
  CheckRandnStatistics(RunRandnSeedOffset<half>(seed, offset));
  CheckRandnStatistics(RunRandnSeedOffset<bfloat16_t>(seed, offset));
}

TEST(TestApiRandSeedOffset, TensorSeedEqualsScalarSeed) {
  const int64_t seed = 987654321;
  const std::vector<int32_t> offset = {0, 1, 2, 3, 17, 128};
  const auto tensorSeedResult = RunRandnSeedOffset<uint32_t>(seed, offset);
  const auto scalarSeedResult = RunRandnScalarSeedOffset<uint32_t>(seed, offset);
  EXPECT_EQ(tensorSeedResult, scalarSeedResult);
}

TEST(TestApiRandSeedOffset, SameSeedAndOffsetIsDeterministic) {
  const int64_t seed = 7;
  const std::vector<int32_t> offset = {1, 2, 3, 4};
  const auto first = RunRandnSeedOffset<uint32_t>(seed, offset);
  const auto second = RunRandnSeedOffset<uint32_t>(seed, offset);
  EXPECT_EQ(first, second);
  const auto otherSeed = RunRandnSeedOffset<uint32_t>(seed + 1, offset);
  EXPECT_NE(first, otherSeed);
}

TEST(TestApiRandSeedOffset, DifferentOffsetsGenerateDifferentOutputs) {
  const int64_t seed = 7;
  const std::vector<int32_t> offset = {0, 1, 2, 3};
  const auto first = RunRandnSeedOffset<uint32_t>(seed, offset);
  const auto second = RunRandnSeedOffset<uint32_t>(seed + 1, offset);
  const std::vector<int32_t> otherOffset = {4, 5, 6, 7};
  const auto otherOffsetResult = RunRandnSeedOffset<uint32_t>(seed, otherOffset);
  EXPECT_NE(first, otherOffsetResult);
  for (size_t i = 0U; i < offset.size(); ++i) {
    EXPECT_NE(first[i], second[i]);
    EXPECT_NE(first[i], otherOffsetResult[i]);
  }
}
