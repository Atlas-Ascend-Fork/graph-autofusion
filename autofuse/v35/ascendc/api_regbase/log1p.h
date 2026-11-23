/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_LOG1P_H__
#define __ASCENDC_API_LOG1P_H__

namespace AscendC {
namespace Log1pAPI {

constexpr Reg::CastTrait CAST_B16_TO_F32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::UNKNOWN};
constexpr Reg::CastTrait CAST_F32_TO_B16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::CAST_RINT};

template <typename T>
__simd_vf__ inline void Log1pCompute(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount) {
  static_assert(SupportType<T, float, half, bfloat16_t>(), "Log1p requires float, half or bfloat16_t");
  constexpr uint32_t oneRepSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));
  uint16_t repeatTimes = CeilDivision(calCount, oneRepSize);

  constexpr float positiveInf = __builtin_inff();
  Reg::RegTensor<float> srcReg, sumReg, correctionReg, resultReg;
  Reg::MaskReg mask, cmpMask;

  uint32_t sreg = calCount;
  for (uint16_t i = 0; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<float>(sreg);

    if constexpr (std::is_same_v<T, float>) {
      Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(srcReg, src + i * oneRepSize);
    } else {
      Reg::RegTensor<T> inputB16;
      Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputB16, src + i * oneRepSize);
      Reg::Cast<float, T, CAST_B16_TO_F32>(srcReg, inputB16, mask);
    }

    Reg::Adds(sumReg, srcReg, 1.0f, mask);
    Reg::Adds(correctionReg, sumReg, -1.0f, mask);
    Reg::Div(correctionReg, srcReg, correctionReg, mask);
    Reg::Log(resultReg, sumReg, mask);
    Reg::Mul(resultReg, resultReg, correctionReg, mask);
    Reg::Compares<float, CMPMODE::NE>(cmpMask, sumReg, 1.0f, mask);
    Reg::Select(resultReg, resultReg, srcReg, cmpMask);
    Reg::Compares<float, CMPMODE::NE>(cmpMask, sumReg, positiveInf, mask);
    Reg::Duplicate(correctionReg, positiveInf, mask);
    Reg::Select(resultReg, resultReg, correctionReg, cmpMask);

    if constexpr (std::is_same_v<T, float>) {
      Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(dst + i * oneRepSize, resultReg, mask);
    } else {
      Reg::RegTensor<T> outputB16;
      Reg::Cast<T, float, CAST_F32_TO_B16>(outputB16, resultReg, mask);
      Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, outputB16, mask);
    }
  }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void Log1pImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                 const uint32_t calCount) {
  if ASCEND_IS_AIC {
    return;
  }

  Log1pAPI::Log1pCompute((__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(), calCount);
}

}  // namespace Log1pAPI
}  // namespace AscendC

template <typename T>
inline __aicore__ void Log1p(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                             const uint32_t calCount) {
  AscendC::Log1pAPI::Log1pImpl(dst, src, calCount);
}
#endif  // __ASCENDC_API_LOG1P_H__
