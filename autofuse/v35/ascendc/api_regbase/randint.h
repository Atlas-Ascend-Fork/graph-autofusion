/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_RANDINT_H__
#define __ASCENDC_API_REGBASE_RANDINT_H__

// RandInt: 逐元素均匀整数随机数 y[i] = low[i] + Philox64(seed[i], counter[i]) % (high[i] - low[i])
// 合同（与 torch inductor randint64 一致）：
//   x1 seed：int64，运行时标量广播（也可为编译期常量）
//   x2 counter：int32，逐元素位置流（Arange 产出）
//   x3 low / x4 high：int64，编译期标量广播
// key 取 seed 低/高 32 位，counter 低 32 位取 x2[i] 升位，其余为 0；
// 取 Philox4x32-10 输出前两个 uint32 拼 uint64 后对 (high - low) 取模。
constexpr uint32_t RANDINT_THREAD_NUM = 1024;
constexpr uint32_t RANDINT_PHILOX_MUL0 = 0xD2511F53u;
constexpr uint32_t RANDINT_PHILOX_MUL1 = 0xCD9E8D57u;
constexpr uint32_t RANDINT_PHILOX_KEY_ADD0 = 0x9E3779B9u;
constexpr uint32_t RANDINT_PHILOX_KEY_ADD1 = 0xBB67AE85u;
constexpr uint32_t RANDINT_PHILOX_ROUNDS = 10;
constexpr uint32_t RANDINT_PHILOX_COUNTER_NUM = 4;  // Philox4x32 counter/输出元素数（4 × 32bit）

__simt_callee__ __aicore__ inline void RandIntPhiloxRound(uint32_t *counter, const uint32_t *key) {
  const uint64_t res0 = static_cast<uint64_t>(RANDINT_PHILOX_MUL0) * static_cast<uint64_t>(counter[0]);
  const uint64_t res1 = static_cast<uint64_t>(RANDINT_PHILOX_MUL1) * static_cast<uint64_t>(counter[2]);
  const uint32_t res0Lo = static_cast<uint32_t>(res0 & 0xFFFFFFFFu);
  const uint32_t res0Hi = static_cast<uint32_t>(res0 >> 32U);
  const uint32_t res1Lo = static_cast<uint32_t>(res1 & 0xFFFFFFFFu);
  const uint32_t res1Hi = static_cast<uint32_t>(res1 >> 32U);
  counter[0] = res1Hi ^ counter[1] ^ key[0];
  counter[1] = res1Lo;
  counter[2] = res0Hi ^ counter[3] ^ key[1];
  counter[3] = res0Lo;
}

// 10 轮迭代使用临时变量，不修改传入的 key/counter
__simt_callee__ __aicore__ inline void RandIntPhilox4x32(const uint32_t *key, const uint32_t *counter,
                                                         uint32_t *results) {
  uint32_t keyTmp[2] = {key[0], key[1]};
  uint32_t counterTmp[RANDINT_PHILOX_COUNTER_NUM] = {counter[0], counter[1], counter[2], counter[3]};
#pragma unroll
  for (uint32_t i = 0; i < RANDINT_PHILOX_ROUNDS; i++) {
    RandIntPhiloxRound(counterTmp, keyTmp);
    keyTmp[0] += RANDINT_PHILOX_KEY_ADD0;
    keyTmp[1] += RANDINT_PHILOX_KEY_ADD1;
  }
#pragma unroll
  for (uint32_t i = 0; i < RANDINT_PHILOX_COUNTER_NUM; i++) {
    results[i] = counterTmp[i];
  }
}

// 标量场景的输入以标量值传入（seedScalar/lowScalar/highScalar），非标量场景从 UB 逐元素读取
template <typename T, bool kSeedIsScalar, bool kLowIsScalar, bool kHighIsScalar>
__simt_vf__ __aicore__ LAUNCH_BOUND(RANDINT_THREAD_NUM) inline void RandIntSimtCompute(
    __ubuf__ T *dst, __ubuf__ int64_t *seed, __ubuf__ int32_t *counter, __ubuf__ int64_t *low, __ubuf__ int64_t *high,
    int64_t seedScalar, int64_t lowScalar, int64_t highScalar, const int64_t total_num) {
  for (int64_t i = threadIdx.x; i < total_num; i += blockDim.x) {
    const int64_t seedVal = kSeedIsScalar ? seedScalar : seed[i];
    const int64_t lowVal = kLowIsScalar ? lowScalar : low[i];
    const int64_t highVal = kHighIsScalar ? highScalar : high[i];
    const uint32_t key[2] = {static_cast<uint32_t>(static_cast<uint64_t>(seedVal) & 0xFFFFFFFFu),
                             static_cast<uint32_t>(static_cast<uint64_t>(seedVal) >> 32U)};
    const uint32_t counterTmp[RANDINT_PHILOX_COUNTER_NUM] = {static_cast<uint32_t>(counter[i]), 0, 0, 0};
    uint32_t results[RANDINT_PHILOX_COUNTER_NUM] = {0, 0, 0, 0};
    RandIntPhilox4x32(key, counterTmp, results);
    const uint64_t randU64 = (static_cast<uint64_t>(results[0]) << 32U) | static_cast<uint64_t>(results[1]);
    const uint64_t range = static_cast<uint64_t>(highVal) - static_cast<uint64_t>(lowVal);
    dst[i] = static_cast<T>(lowVal + static_cast<int64_t>(randU64 % range));
  }
}

template <typename T>
__aicore__ inline void RandInt(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &seed,
                               const AscendC::LocalTensor<int32_t> &counter, const AscendC::LocalTensor<T> &low,
                               const AscendC::LocalTensor<T> &high, const AscendC::LocalTensor<uint8_t> &tmp_buf,
                               const uint32_t size) {
  AscendC::Simt::VF_CALL<RandIntSimtCompute<T, false, false, false>>(
      AscendC::Simt::Dim3(RANDINT_THREAD_NUM), (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ int64_t *)seed.GetPhyAddr(),
      (__ubuf__ int32_t *)counter.GetPhyAddr(), (__ubuf__ int64_t *)low.GetPhyAddr(),
      (__ubuf__ int64_t *)high.GetPhyAddr(), 0, 0, 0, size);
}

template <typename T>
__aicore__ inline void RandInt(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &seed,
                               const AscendC::LocalTensor<int32_t> &counter, const T &low, const T &high,
                               const AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  AscendC::Simt::VF_CALL<RandIntSimtCompute<T, false, true, true>>(
      AscendC::Simt::Dim3(RANDINT_THREAD_NUM), (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ int64_t *)seed.GetPhyAddr(),
      (__ubuf__ int32_t *)counter.GetPhyAddr(), (__ubuf__ int64_t *)nullptr, (__ubuf__ int64_t *)nullptr, 0,
      static_cast<int64_t>(low), static_cast<int64_t>(high), size);
}

template <typename T>
__aicore__ inline void RandInt(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &seed,
                               const AscendC::LocalTensor<int32_t> &counter, const AscendC::LocalTensor<T> &low,
                               const T &high, const AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  AscendC::Simt::VF_CALL<RandIntSimtCompute<T, false, false, true>>(
      AscendC::Simt::Dim3(RANDINT_THREAD_NUM), (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ int64_t *)seed.GetPhyAddr(),
      (__ubuf__ int32_t *)counter.GetPhyAddr(), (__ubuf__ int64_t *)low.GetPhyAddr(), (__ubuf__ int64_t *)nullptr, 0, 0,
      static_cast<int64_t>(high), size);
}

template <typename T>
__aicore__ inline void RandInt(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &seed,
                               const AscendC::LocalTensor<int32_t> &counter, const T &low,
                               const AscendC::LocalTensor<T> &high, const AscendC::LocalTensor<uint8_t> &tmp_buf,
                               const uint32_t size) {
  AscendC::Simt::VF_CALL<RandIntSimtCompute<T, false, true, false>>(
      AscendC::Simt::Dim3(RANDINT_THREAD_NUM), (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ int64_t *)seed.GetPhyAddr(),
      (__ubuf__ int32_t *)counter.GetPhyAddr(), (__ubuf__ int64_t *)nullptr, (__ubuf__ int64_t *)high.GetPhyAddr(), 0,
      static_cast<int64_t>(low), 0, size);
}

template <typename T>
__aicore__ inline void RandInt(const AscendC::LocalTensor<T> &dst, const T &seed,
                               const AscendC::LocalTensor<int32_t> &counter, const AscendC::LocalTensor<T> &low,
                               const AscendC::LocalTensor<T> &high, const AscendC::LocalTensor<uint8_t> &tmp_buf,
                               const uint32_t size) {
  AscendC::Simt::VF_CALL<RandIntSimtCompute<T, true, false, false>>(
      AscendC::Simt::Dim3(RANDINT_THREAD_NUM), (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ int64_t *)nullptr,
      (__ubuf__ int32_t *)counter.GetPhyAddr(), (__ubuf__ int64_t *)low.GetPhyAddr(),
      (__ubuf__ int64_t *)high.GetPhyAddr(), static_cast<int64_t>(seed), 0, 0, size);
}

template <typename T>
__aicore__ inline void RandInt(const AscendC::LocalTensor<T> &dst, const T &seed,
                               const AscendC::LocalTensor<int32_t> &counter, const T &low, const T &high,
                               const AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  AscendC::Simt::VF_CALL<RandIntSimtCompute<T, true, true, true>>(
      AscendC::Simt::Dim3(RANDINT_THREAD_NUM), (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ int64_t *)nullptr,
      (__ubuf__ int32_t *)counter.GetPhyAddr(), (__ubuf__ int64_t *)nullptr, (__ubuf__ int64_t *)nullptr,
      static_cast<int64_t>(seed), static_cast<int64_t>(low), static_cast<int64_t>(high), size);
}

template <typename T>
__aicore__ inline void RandInt(const AscendC::LocalTensor<T> &dst, const T &seed,
                               const AscendC::LocalTensor<int32_t> &counter, const AscendC::LocalTensor<T> &low,
                               const T &high, const AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  AscendC::Simt::VF_CALL<RandIntSimtCompute<T, true, false, true>>(
      AscendC::Simt::Dim3(RANDINT_THREAD_NUM), (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ int64_t *)nullptr,
      (__ubuf__ int32_t *)counter.GetPhyAddr(), (__ubuf__ int64_t *)low.GetPhyAddr(), (__ubuf__ int64_t *)nullptr,
      static_cast<int64_t>(seed), 0, static_cast<int64_t>(high), size);
}

template <typename T>
__aicore__ inline void RandInt(const AscendC::LocalTensor<T> &dst, const T &seed,
                               const AscendC::LocalTensor<int32_t> &counter, const T &low,
                               const AscendC::LocalTensor<T> &high, const AscendC::LocalTensor<uint8_t> &tmp_buf,
                               const uint32_t size) {
  AscendC::Simt::VF_CALL<RandIntSimtCompute<T, true, true, false>>(
      AscendC::Simt::Dim3(RANDINT_THREAD_NUM), (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ int64_t *)nullptr,
      (__ubuf__ int32_t *)counter.GetPhyAddr(), (__ubuf__ int64_t *)nullptr, (__ubuf__ int64_t *)high.GetPhyAddr(),
      static_cast<int64_t>(seed), static_cast<int64_t>(low), 0, size);
}

#endif  // __ASCENDC_API_REGBASE_RANDINT_H__
