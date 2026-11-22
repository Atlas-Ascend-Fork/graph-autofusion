/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. This file is provided on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 */
#ifndef __ASCENDC_API_REGBASE_SIGMOID_H__
#define __ASCENDC_API_REGBASE_SIGMOID_H__

namespace AscendC {
namespace SigmoidAPI {

constexpr Reg::CastTrait CAST_B16_TO_F32 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
constexpr Reg::CastTrait CAST_F32_TO_B16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};
constexpr uint32_t SIGMOID_FLOAT_NAN = 0x7fc00000;
constexpr uint32_t SIGMOID_FLOAT_POS_INF = 0x7f800000;
// 正侧饱和阈值：x ≥ 20 时 1/(1+e^-20) 在 fp32/half/bfloat16 下均舍入为 1.0，饱和无精度损失。
// 同时绕开 Reg::Exp 在 e^-x 落入 fp32 次正规区间（x ≳ 87，--cce-ftz=false）时误返回 ~1、
// 致 sigmoid(x)=1/(1+1)=0.5 的错误路径；负侧 exp(-x) 为正规大数或 inf，不在饱和范围。
constexpr float SIGMOID_SATURATION_THRESHOLD = 20.0f;
// Keep the sigmoid sequence aligned with the register-level VF implementation:
// x -> -x -> exp(-x) -> 1 + exp(-x) -> 1 / (1 + exp(-x)).
template <typename T>
__simd_vf__ inline void SigmoidCompute(__ubuf__ T *dstUb, __ubuf__ T *srcUb, uint32_t count,
                                       const uint16_t repeatTimes) {
  uint32_t sreg = count;
  Reg::MaskReg preg;
  Reg::RegTensor<T> srcReg;
  Reg::RegTensor<float> castReg;
  Reg::RegTensor<float> tmpReg;
  Reg::RegTensor<float> dstReg;
  Reg::RegTensor<float> nanReg;
  Reg::RegTensor<float> specialReg;
  Reg::RegTensor<float> oneReg;
  Reg::RegTensor<float> zeroReg;
  Reg::MaskReg nanMask;
  Reg::MaskReg posInfMask;
  Reg::MaskReg negInfMask;
  Reg::MaskReg satMask;

  for (uint16_t i = 0; i < repeatTimes; ++i) {
    preg = Reg::UpdateMask<float>(sreg);
    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
      Reg::Cast<float, T, CAST_B16_TO_F32>(castReg, srcReg, preg);
    } else {
      Reg::LoadAlign(castReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
    }

    Reg::Compare<float, CMPMODE::NE>(nanMask, castReg, castReg, preg);
    Reg::Duplicate(specialReg, (float &)SIGMOID_FLOAT_POS_INF, preg);
    Reg::Compare<float, CMPMODE::EQ>(posInfMask, castReg, specialReg, preg);
    Reg::Neg(specialReg, specialReg, preg);
    Reg::Compare<float, CMPMODE::EQ>(negInfMask, castReg, specialReg, preg);
    // 大正 x 饱和：NaN 与阈值比较恒为 false ⇒ 走通用路径，不影响 NaN 传播。
    Reg::CompareScalar<float, CMPMODE::GE>(satMask, castReg, SIGMOID_SATURATION_THRESHOLD, preg);

    Reg::Muls(tmpReg, castReg, -1.0f, preg);
    Reg::Exp(tmpReg, tmpReg, preg);
    Reg::Adds(tmpReg, tmpReg, 1.0f, preg);
    Reg::Duplicate(dstReg, 1.0f, preg);
    Reg::Div(dstReg, dstReg, tmpReg, preg);

    Reg::Duplicate(nanReg, (float &)SIGMOID_FLOAT_NAN, preg);
    Reg::Select(dstReg, nanReg, dstReg, nanMask);
    Reg::Duplicate(oneReg, 1.0f, preg);
    Reg::Select(dstReg, oneReg, dstReg, posInfMask);
    Reg::Select(dstReg, oneReg, dstReg, satMask);
    Reg::Duplicate(zeroReg, 0.0f, preg);
    Reg::Select(dstReg, zeroReg, dstReg, negInfMask);

    if constexpr (sizeof(T) == sizeof(half)) {
      Reg::Cast<T, float, CAST_F32_TO_B16>(srcReg, dstReg, preg);
      Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dstUb + i * B32_DATA_NUM_PER_REPEAT, srcReg, preg);
    } else {
      Reg::StoreAlign(dstUb + i * B32_DATA_NUM_PER_REPEAT, dstReg, preg);
    }
  }
}

}  // namespace SigmoidAPI
}  // namespace AscendC

/**
 * @brief SigmoidExtend - compute the sigmoid activation: y = 1 / (1 + exp(-x))
 *
 * Uses register-level SIMD with float intermediates for float, half, and bfloat16 inputs.
 *
 * @tparam T data type, supports float, half, and bfloat16_t
 * @param dst output tensor
 * @param src input tensor
 * @param tmpBuf temporary buffer (unused; kept for API compatibility)
 * @param calCount number of elements to compute
 */
template <typename T>
__aicore__ inline void SigmoidExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                     AscendC::LocalTensor<uint8_t> &tmpBuf, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, half, float, bfloat16_t>(),
                "SigmoidExtend only supports half, float and bfloat16_t on current device!");
  (void)tmpBuf;

  constexpr uint32_t oneRepSize = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
  const uint16_t repeatTimes = AscendC::CeilDivision(calCount, oneRepSize);
  AscendC::SigmoidAPI::SigmoidCompute((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), calCount,
                                      repeatTimes);
}

#endif  // __ASCENDC_API_REGBASE_SIGMOID_H__
