/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_RANDOM_H__
#define __ASCENDC_API_REGBASE_RANDOM_H__

constexpr uint32_t RAND_THREAD_NUM = 1024U;
constexpr uint32_t RAND_PHILOX_MUL0 = 0xD2511F53U;
constexpr uint32_t RAND_PHILOX_MUL1 = 0xCD9E8D57U;
constexpr uint32_t RAND_PHILOX_KEY_ADD0 = 0x9E3779B9U;
constexpr uint32_t RAND_PHILOX_KEY_ADD1 = 0xBB67AE85U;
constexpr uint32_t RAND_PHILOX_ROUNDS = 10U;
constexpr float RAND_NORMAL_MIN_UNIFORM = 1e-7F;
constexpr float RAND_TWO_PI = 6.283185307179586F;

template <typename T>
__simt_callee__ __aicore__ inline T RandConvertFloat(const float value) {
  return static_cast<T>(value);
}

template <>
__simt_callee__ __aicore__ inline bfloat16_t RandConvertFloat<bfloat16_t>(const float value) {
  union FloatBits {
    uint32_t bits;
    float value;
  };
  constexpr uint32_t kBfloat16RoundBias = 0x7FFFU;
  FloatBits floatBits = {};
  floatBits.value = value;
  const uint32_t rounded = floatBits.bits + kBfloat16RoundBias + ((floatBits.bits >> 16U) & 1U);
  union Bfloat16Bits {
    uint16_t bits;
    bfloat16_t value;
  };
  const Bfloat16Bits bfloat16_bits = {static_cast<uint16_t>(rounded >> 16U)};
  return bfloat16_bits.value;
}

template <typename T>
__simt_callee__ __aicore__ inline T RandConvertUniform(const float value) {
  T result = RandConvertFloat<T>(value);
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    if (static_cast<float>(result) >= 1.0F) {
      result = static_cast<T>(0.99609375F);
    }
  } else if constexpr (std::is_same_v<T, half>) {
    if (static_cast<float>(result) >= 1.0F) {
      result = static_cast<T>(0.99951171875F);
    }
  }
  return result;
}

__simt_callee__ __aicore__ inline float RandUniform(const uint32_t value) {
  union FloatBits {
    uint32_t bits;
    float value;
  };
  constexpr uint32_t kMantissaMask = 0x7FFFFFU;
  constexpr uint32_t kExponentMask = 127U << 23U;
  const FloatBits float_bits = {(value & kMantissaMask) | kExponentMask};
  return float_bits.value - 1.0F;
}

__simt_callee__ __aicore__ inline void RandPhiloxRound(uint32_t *counter, const uint32_t *key) {
  const uint64_t res0 = static_cast<uint64_t>(RAND_PHILOX_MUL0) * static_cast<uint64_t>(counter[0]);
  const uint64_t res1 = static_cast<uint64_t>(RAND_PHILOX_MUL1) * static_cast<uint64_t>(counter[2]);
  const uint32_t res0Lo = static_cast<uint32_t>(res0 & 0xFFFFFFFFU);
  const uint32_t res0Hi = static_cast<uint32_t>(res0 >> 32U);
  const uint32_t res1Lo = static_cast<uint32_t>(res1 & 0xFFFFFFFFU);
  const uint32_t res1Hi = static_cast<uint32_t>(res1 >> 32U);
  counter[0] = res1Hi ^ counter[1] ^ key[0];
  counter[1] = res1Lo;
  counter[2] = res0Hi ^ counter[3] ^ key[1];
  counter[3] = res0Lo;
}

__simt_callee__ __aicore__ inline void RandPhilox4x32(const uint32_t *key, const uint32_t *counter, uint32_t *results) {
  uint32_t keyTmp[2] = {key[0], key[1]};
  uint32_t counterTmp[4] = {counter[0], counter[1], counter[2], counter[3]};
#pragma unroll
  for (uint32_t i = 0; i < RAND_PHILOX_ROUNDS; i++) {
    RandPhiloxRound(counterTmp, keyTmp);
    keyTmp[0] += RAND_PHILOX_KEY_ADD0;
    keyTmp[1] += RAND_PHILOX_KEY_ADD1;
  }
#pragma unroll
  for (uint32_t i = 0; i < 4U; i++) {
    results[i] = counterTmp[i];
  }
}

template <typename T, bool kSeedIsScalar, bool kIsRandn>
__simt_vf__ __aicore__ LAUNCH_BOUND(RAND_THREAD_NUM) inline void RandSeedOffsetSimtCompute(
    __ubuf__ T *dst, __ubuf__ int64_t *seed, __ubuf__ int32_t *offset, int64_t seedScalar, const int64_t total_num) {
  for (int64_t i = threadIdx.x; i < total_num; i += blockDim.x) {
    const int64_t seedVal = kSeedIsScalar ? seedScalar : seed[i];
    const uint32_t key[2] = {static_cast<uint32_t>(static_cast<uint64_t>(seedVal) & 0xFFFFFFFFU),
                             static_cast<uint32_t>(static_cast<uint64_t>(seedVal) >> 32U)};
    const uint32_t counter[4] = {static_cast<uint32_t>(offset[i]), 0U, 0U, 0U};
    uint32_t results[4] = {0U, 0U, 0U, 0U};
    RandPhilox4x32(key, counter, results);
    if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) {
      dst[i] = static_cast<T>(results[0]);
    } else if constexpr (kIsRandn) {
      const float uniform = RandUniform(results[0]);
      const float clamped_uniform = uniform < RAND_NORMAL_MIN_UNIFORM ? RAND_NORMAL_MIN_UNIFORM : uniform;
      const float uniform_x = AscendC::Simt::Sqrt(-2.0F * AscendC::Simt::Log(clamped_uniform));
      const float uniform_y = AscendC::Simt::Cos(RAND_TWO_PI * RandUniform(results[1]));
      dst[i] = RandConvertFloat<T>(uniform_x * uniform_y);
    } else {
      dst[i] = RandConvertUniform<T>(RandUniform(results[0]));
    }
  }
}

template <typename T, bool kSeedIsScalar, bool kIsRandn>
__aicore__ inline void RandSeedOffsetImpl(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<int64_t> &seed,
                                          const int64_t seedScalar, const AscendC::LocalTensor<int32_t> &offset,
                                          const AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t size) {
  (void)tmpBuf;
  AscendC::Simt::VF_CALL<RandSeedOffsetSimtCompute<T, kSeedIsScalar, kIsRandn>>(
      AscendC::Simt::Dim3(RAND_THREAD_NUM), (__ubuf__ T *)dst.GetPhyAddr(),
      kSeedIsScalar ? (__ubuf__ int64_t *)nullptr : (__ubuf__ int64_t *)seed.GetPhyAddr(),
      (__ubuf__ int32_t *)offset.GetPhyAddr(), seedScalar, static_cast<int64_t>(size));
}

template <typename T, bool kIsRandn>
__simt_vf__ __aicore__ LAUNCH_BOUND(RAND_THREAD_NUM) inline void RandFixedSimtCompute(__ubuf__ T *dst,
                                                                                      const int64_t total_num) {
  for (int64_t i = threadIdx.x; i < total_num; i += blockDim.x) {
    const uint32_t key[2] = {0U, 0U};
    const uint32_t counter[4] = {static_cast<uint32_t>(i), 0U, 0U, 0U};
    uint32_t results[4] = {0U, 0U, 0U, 0U};
    RandPhilox4x32(key, counter, results);
    if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) {
      dst[i] = static_cast<T>(results[0]);
    } else if constexpr (kIsRandn) {
      const float uniform = RandUniform(results[0]);
      const float clamped_uniform = uniform < RAND_NORMAL_MIN_UNIFORM ? RAND_NORMAL_MIN_UNIFORM : uniform;
      const float uniform_x = AscendC::Simt::Sqrt(-2.0F * AscendC::Simt::Log(clamped_uniform));
      const float uniform_y = AscendC::Simt::Cos(RAND_TWO_PI * RandUniform(results[1]));
      dst[i] = RandConvertFloat<T>(uniform_x * uniform_y);
    } else {
      dst[i] = RandConvertUniform<T>(RandUniform(results[0]));
    }
  }
}

/**
 * @brief Rand - unified random number generation using AscendC PhiloxRandom API
 * Uses fixed philoxKey = {0, 0} and philoxCounter = {0, 0, 0, 0} for continuous mode
 * Supports uint32_t, int32_t, half, bfloat16_t and float data types
 *
 * @tparam T data type, supports uint32_t, int32_t, half, bfloat16_t and float
 * @param dst output tensor
 * @param size number of elements to generate
 */
template <typename T>
inline __aicore__ void Rand(const AscendC::LocalTensor<T> &dst, const uint32_t size) {
  if constexpr (std::is_same_v<T, float>) {
    AscendC::PhiloxKey philoxKey = {0, 0};
    AscendC::PhiloxCounter philoxCounter = {0, 0, 0, 0};
    AscendC::PhiloxRandom<10>(dst, philoxKey, philoxCounter, size);
  } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
    AscendC::Simt::VF_CALL<RandFixedSimtCompute<T, false>>(AscendC::Simt::Dim3(RAND_THREAD_NUM),
                                                           (__ubuf__ T *)dst.GetPhyAddr(), static_cast<int64_t>(size));
  } else {
    AscendC::PhiloxKey philoxKey = {0, 0};
    AscendC::PhiloxCounter philoxCounter = {0, 0, 0, 0};
    AscendC::PhiloxRandom<10>(dst, philoxKey, philoxCounter, size);
  }
}

/**
 * @brief Randn - generate fixed-seed normal random numbers
 * Uses key={0, 0} and offset={i, 0, 0, 0}. uint32_t/int32_t emit raw Philox
 * words, while floating-point types use the SIMT normal path.
 */
template <typename T>
inline __aicore__ void Randn(const AscendC::LocalTensor<T> &dst, const uint32_t size) {
  if constexpr (!std::is_same_v<T, uint32_t> && !std::is_same_v<T, int32_t>) {
    AscendC::Simt::VF_CALL<RandFixedSimtCompute<T, true>>(AscendC::Simt::Dim3(RAND_THREAD_NUM),
                                                          (__ubuf__ T *)dst.GetPhyAddr(), static_cast<int64_t>(size));
    return;
  }
  Rand(dst, size);
}

template <typename T>
__aicore__ inline void RandSeedOffset(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<int64_t> &seed,
                                      const AscendC::LocalTensor<int32_t> &offset,
                                      const AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t size) {
  RandSeedOffsetImpl<T, false, false>(dst, seed, 0, offset, tmpBuf, size);
}

template <typename T>
__aicore__ inline void RandSeedOffset(const AscendC::LocalTensor<T> &dst, const int64_t seed,
                                      const AscendC::LocalTensor<int32_t> &offset,
                                      const AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t size) {
  AscendC::LocalTensor<int64_t> emptySeed;
  RandSeedOffsetImpl<T, true, false>(dst, emptySeed, seed, offset, tmpBuf, size);
}

template <typename T>
__aicore__ inline void RandnSeedOffset(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<int64_t> &seed,
                                       const AscendC::LocalTensor<int32_t> &offset,
                                       const AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t size) {
  RandSeedOffsetImpl<T, false, true>(dst, seed, 0, offset, tmpBuf, size);
}

template <typename T>
__aicore__ inline void RandnSeedOffset(const AscendC::LocalTensor<T> &dst, const int64_t seed,
                                       const AscendC::LocalTensor<int32_t> &offset,
                                       const AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t size) {
  AscendC::LocalTensor<int64_t> emptySeed;
  RandSeedOffsetImpl<T, true, true>(dst, emptySeed, seed, offset, tmpBuf, size);
}

#endif  // __ASCENDC_API_REGBASE_RANDOM_H__
