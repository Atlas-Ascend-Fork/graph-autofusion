/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_RANDN_H__
#define __ASCENDC_API_REGBASE_RANDN_H__

#include "adv_api/math/philox.h"
#include "trigonometric_function_utils.h"

namespace AscendC {

enum class RandnPhiloxRounds : uint16_t {
  ROUNDS_7 = 7,
  ROUNDS_10 = 10,
};

constexpr uint32_t PHILOX_NORMAL_RANDOM_MAX_COUNT = 65535U;
constexpr uint32_t PHILOX_NORMAL_RANDOM_BLOCK_COUNT = 4U;
constexpr uint32_t PHILOX_NORMAL_RANDOM_UINT32_PER_REPEAT = B32_DATA_NUM_PER_REPEAT;
constexpr float PHILOX_NORMAL_RANDOM_TWO_PI = 6.28318530717958647692f;
constexpr float PHILOX_NORMAL_RANDOM_MIN_UNIFORM = 1.0e-7f;

template <bool PairAligned>
__simd_vf__ inline void PhiloxNormalRandomCoreImpl(__ubuf__ float *dstUb, __ubuf__ float *srcUb, uint32_t pairCount,
                                                   uint16_t repeatTimes) {
  Reg::MaskReg mask;
  Reg::UnalignReg loadUreg, storeUreg;
  Reg::RegTensor<float> u1Reg, u2Reg, sinReg, cosReg;
#pragma unroll 3
  for (uint16_t i = 0U; i < repeatTimes; ++i) {
    uint32_t offset = static_cast<uint32_t>(i) * PHILOX_NORMAL_RANDOM_UINT32_PER_REPEAT;
    uint32_t currentCount = pairCount - offset;
    if (currentCount > PHILOX_NORMAL_RANDOM_UINT32_PER_REPEAT) {
      currentCount = PHILOX_NORMAL_RANDOM_UINT32_PER_REPEAT;
    }
    mask = Reg::UpdateMask<float>(currentCount);
    Reg::LoadAlign(u1Reg, srcUb + offset);
    __ubuf__ float *srcUb2 = srcUb + pairCount + offset;
    if constexpr (PairAligned) {
      Reg::LoadAlign(u2Reg, srcUb2);
    } else {
      Reg::LoadUnAlignPre(loadUreg, srcUb2);
      Reg::LoadUnAlign(u2Reg, loadUreg, srcUb2, currentCount);
    }
    Reg::Maxs(u1Reg, u1Reg, PHILOX_NORMAL_RANDOM_MIN_UNIFORM, mask);
    Reg::Log(u1Reg, u1Reg, mask);
    Reg::Muls(u1Reg, u1Reg, -2.0f, mask);
    Reg::Sqrt(u1Reg, u1Reg, mask);
    Reg::Muls(u2Reg, u2Reg, PHILOX_NORMAL_RANDOM_TWO_PI, mask);
    AutofuseCos(cosReg, u2Reg, mask);
    AutofuseSin(sinReg, u2Reg, mask);
    Reg::Mul(u2Reg, u1Reg, sinReg, mask);
    Reg::Mul(u1Reg, u1Reg, cosReg, mask);
    Reg::StoreAlign(dstUb + offset, u1Reg, mask);
    __ubuf__ float *dstUb2 = dstUb + pairCount + offset;
    if constexpr (PairAligned) {
      Reg::StoreAlign(dstUb2, u2Reg, mask);
    } else {
      Reg::StoreUnAlign(dstUb2, u2Reg, storeUreg, currentCount);
      Reg::StoreUnAlignPost(dstUb2, storeUreg, 0);
    }
  }
}

template <typename T, RandnPhiloxRounds Rounds = RandnPhiloxRounds::ROUNDS_10>
__aicore__ inline void RandnExtend(const LocalTensor<T> &dst, const LocalTensor<T> &uniformTmp,
                                   const PhiloxKey &philoxKey, const PhiloxCounter &philoxCounter, uint16_t count) {
  static_assert(std::is_same_v<T, float>, "RandnExtend only supports float currently!");
  static_assert(Rounds == RandnPhiloxRounds::ROUNDS_7 || Rounds == RandnPhiloxRounds::ROUNDS_10,
                "RandnExtend only supports 7 or 10 Philox rounds!");
  if ASCEND_IS_AIC {
    return;
  }

  if (count == 0U) {
    return;
  }

  ASCENDC_ASSERT(((count % PHILOX_NORMAL_RANDOM_BLOCK_COUNT) == 0U), {
    KERNEL_LOG(KERNEL_ERROR, "RandnExtend count must be divisible by %u, got %u", PHILOX_NORMAL_RANDOM_BLOCK_COUNT,
               count);
  });
  ASCENDC_ASSERT((count <= PHILOX_NORMAL_RANDOM_MAX_COUNT), {
    KERNEL_LOG(KERNEL_ERROR, "RandnExtend count should be less than or equal to %u, got %u",
               PHILOX_NORMAL_RANDOM_MAX_COUNT, count);
  });

  __ubuf__ float *dstUb = (__ubuf__ float *)dst.GetPhyAddr();
  __ubuf__ float *srcUb = (__ubuf__ float *)uniformTmp.GetPhyAddr();
  uint32_t pairCount = static_cast<uint32_t>(count / PHILOX_NORMAL_RANDOM_BLOCK_COUNT * 2U);
  uint16_t repeatTimes =
      static_cast<uint16_t>(CeilDivision(pairCount, static_cast<uint32_t>(PHILOX_NORMAL_RANDOM_UINT32_PER_REPEAT)));

  constexpr uint16_t philoxRounds = static_cast<uint16_t>(Rounds);
  PhiloxRandom<philoxRounds, float>(uniformTmp, philoxKey, philoxCounter, count);
  PipeBarrier<PIPE_V>();
  if ((pairCount % (ONE_BLK_SIZE / sizeof(float))) == 0U) {
    PhiloxNormalRandomCoreImpl<true>(dstUb, srcUb, pairCount, repeatTimes);
  } else {
    PhiloxNormalRandomCoreImpl<false>(dstUb, srcUb, pairCount, repeatTimes);
  }
}

}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_RANDN_H__
