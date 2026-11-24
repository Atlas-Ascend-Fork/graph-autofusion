/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_POLYGAMMA_H__
#define __ASCENDC_API_REGBASE_POLYGAMMA_H__

namespace AscendC {
namespace PolyGammaAPI {

constexpr float TRI_PI = 3.14159265358979323846f;
constexpr float TRI_ONE_SIXTH = 1.0f / 6.0f;
constexpr float TRI_ONE_THIRTIETH = 1.0f / 30.0f;
constexpr float TRI_ONE_FORTY_SECOND = 1.0f / 42.0f;

template <typename T>
__simd_callee__ inline void TriGammaReflectionSetup(Reg::RegTensor<T> &srcReg, Reg::RegTensor<T> &vecX,
                                                    Reg::RegTensor<T> &vecSign, Reg::RegTensor<T> &vecResult,
                                                    Reg::RegTensor<T> &oneReg, Reg::MaskReg &mask) {
  Reg::RegTensor<T> vecX_reflect, zeroReg, signPos, signNeg;
  Reg::RegTensor<T> vecReflectTerm, vecReflectTermNeg, vecSinX, piSqReg, sinArg;
  Reg::MaskReg mask_lt_05, mask_eq_zero, mask_not_zero, mask_safe_reflect;

  Reg::Duplicate(zeroReg, static_cast<T>(0.0), mask);
  Reg::Duplicate(vecReflectTerm, static_cast<T>(0.0), mask);
  Reg::Duplicate(vecReflectTermNeg, static_cast<T>(0.0), mask);

  // Reflection branch: x < 0.5 → use 1-x
  Reg::Compares<T, CMPMODE::LT>(mask_lt_05, srcReg, static_cast<T>(0.5), mask);
  Reg::Compares<T, CMPMODE::EQ>(mask_eq_zero, srcReg, static_cast<T>(0.0), mask);
  Reg::Not(mask_not_zero, mask_eq_zero, mask);
  Reg::And(mask_safe_reflect, mask_lt_05, mask_not_zero, mask);

  Reg::Sub(vecX_reflect, oneReg, srcReg, mask);
  Reg::Select(vecX, vecX_reflect, srcReg, mask_lt_05);

  Reg::Duplicate(signPos, static_cast<T>(1.0), mask);
  Reg::Muls(signNeg, signPos, static_cast<T>(-1.0), mask);
  Reg::Select(vecSign, signNeg, signPos, mask_lt_05);

  // PI^2 / sin^2(PI * x), safe-reflect-only to avoid div-by-zero at integers
  Reg::Muls(sinArg, srcReg, static_cast<T>(TRI_PI), mask_safe_reflect);
  AutofuseSin<T>(vecSinX, sinArg, mask_safe_reflect);
  Reg::Mul(vecSinX, vecSinX, vecSinX, mask_safe_reflect);
  Reg::Duplicate(piSqReg, static_cast<T>(TRI_PI * TRI_PI));
  Reg::Div(vecReflectTerm, piSqReg, vecSinX, mask_safe_reflect);
  Reg::Muls(vecReflectTermNeg, vecReflectTerm, static_cast<T>(-1.0), mask_safe_reflect);

  // x < 0.5: start at -vecReflectTerm; else 0
  Reg::Select(vecResult, vecReflectTermNeg, zeroReg, mask_lt_05);
}

// One 1/x^2 series step with recurrence x -> x+1
template <typename T>
__simd_callee__ inline void TriGammaSeriesStep(Reg::RegTensor<T> &vecResult, Reg::RegTensor<T> &vecX,
                                               Reg::RegTensor<T> &oneReg, Reg::MaskReg &mask) {
  Reg::RegTensor<T> vecXsq, vecTmp;
  Reg::Mul(vecXsq, vecX, vecX, mask);
  Reg::Div(vecTmp, oneReg, vecXsq, mask);
  Reg::Add(vecResult, vecResult, vecTmp, mask);
  Reg::Adds(vecX, vecX, static_cast<T>(1.0), mask);
}

// Bernoulli tail: result += (1 + 1/(2x) + S) / x, S = ixx*(1/6 + ixx*(-1/30 + ixx/42))
template <typename T>
__simd_callee__ inline void TriGammaBernoulliTail(Reg::RegTensor<T> &vecResult, Reg::RegTensor<T> &vecX,
                                                  Reg::RegTensor<T> &oneReg, Reg::MaskReg &mask) {
  Reg::RegTensor<T> vecXsq, vecIxx, vecT1, halfRecip, tailTerm;
  Reg::Mul(vecXsq, vecX, vecX, mask);
  Reg::Div(vecIxx, oneReg, vecXsq, mask);

  Reg::Duplicate(vecT1, static_cast<T>(TRI_ONE_FORTY_SECOND), mask);
  Reg::Mul(vecT1, vecT1, vecIxx, mask);
  Reg::Adds(vecT1, vecT1, static_cast<T>(-TRI_ONE_THIRTIETH), mask);
  Reg::Mul(vecT1, vecT1, vecIxx, mask);
  Reg::Adds(vecT1, vecT1, static_cast<T>(TRI_ONE_SIXTH), mask);
  Reg::Mul(vecT1, vecT1, vecIxx, mask);

  Reg::Muls(halfRecip, vecX, static_cast<T>(2.0), mask);
  Reg::Div(halfRecip, oneReg, halfRecip, mask);
  Reg::Adds(tailTerm, halfRecip, static_cast<T>(1.0), mask);
  Reg::Add(tailTerm, tailTerm, vecT1, mask);
  Reg::Div(tailTerm, tailTerm, vecX, mask);
  Reg::Add(vecResult, vecResult, tailTerm, mask);
}

template <typename T>
__simd_vf__ inline void TriGammaImplVF(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, vlSize));

  Reg::MaskReg mask;
  Reg::RegTensor<T> srcReg, vecX, vecSign, vecResult, oneReg;

  for (uint16_t i = 0U; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<T>(calCount);
    Reg::LoadAlign(srcReg, src + i * vlSize);
    Reg::Duplicate(oneReg, static_cast<T>(1.0), mask);

    TriGammaReflectionSetup<T>(srcReg, vecX, vecSign, vecResult, oneReg, mask);

    // 6-term fixed-step series
    for (uint16_t iter = 0U; iter < 6U; ++iter) {
      TriGammaSeriesStep<T>(vecResult, vecX, oneReg, mask);
    }

    TriGammaBernoulliTail<T>(vecResult, vecX, oneReg, mask);

    Reg::Mul(vecResult, vecResult, vecSign, mask);
    Reg::StoreAlign(dst + i * vlSize, vecResult, mask);
  }
}

template <typename T>
__simd_vf__ inline void BroadcastScalarImplVF(__ubuf__ T *dst, T scalar, uint32_t calCount) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, vlSize));

  Reg::MaskReg mask;
  Reg::RegTensor<T> dstReg;

  for (uint16_t i = 0U; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<T>(calCount);
    Reg::Duplicate(dstReg, scalar, mask);
    Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

template <typename T>
__simd_vf__ inline void PolyGammaMulCoeffVF(__ubuf__ T *dst, T coeff, uint32_t calCount) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, vlSize));

  Reg::MaskReg mask;
  Reg::RegTensor<T> dstReg;

  for (uint16_t i = 0U; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<T>(calCount);
    Reg::LoadAlign(dstReg, dst + i * vlSize);
    Reg::Muls(dstReg, dstReg, coeff, mask);
    Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

template <typename T>
__simd_vf__ inline void PolyGammaMulVecVF(__ubuf__ T *dst, __ubuf__ T *coeffVec, uint32_t calCount) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, vlSize));

  Reg::MaskReg mask;
  Reg::RegTensor<T> dstReg, coeffReg;

  for (uint16_t i = 0U; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<T>(calCount);
    Reg::LoadAlign(dstReg, dst + i * vlSize);
    Reg::LoadAlign(coeffReg, coeffVec + i * vlSize);
    Reg::Mul(dstReg, dstReg, coeffReg, mask);
    Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

// ============================================================================
// PolyGammaEdgeFixupVF: fix up edge cases after main computation
// - x == 0    → ±∞ (sign = infSign)
// - x == -∞   → NaN if negInfIsNaN, else signed ±∞ (sign = infSign)
// - x == +∞   → handled internally by each computation path
// ============================================================================
template <typename T>
__simd_vf__ inline void PolyGammaEdgeFixupVF(__ubuf__ T *dst, __ubuf__ T *src, T infSign, bool negInfIsNaN,
                                             uint32_t calCount) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, vlSize));

  Reg::MaskReg mask, mask_zero, mask_neg_inf;
  Reg::RegTensor<T> dstReg, srcReg, posInfReg, negInfReg, infReg;

  for (uint16_t i = 0U; i < repeatTimes; ++i) {
    mask = Reg::UpdateMask<T>(calCount);
    Reg::LoadAlign(srcReg, src + i * vlSize);
    Reg::LoadAlign(dstReg, dst + i * vlSize);

    Reg::Duplicate(posInfReg, (float &)F32_INF, mask);

    // x == 0 → signed ±∞
    Reg::Compares<T, CMPMODE::EQ>(mask_zero, srcReg, static_cast<T>(0.0), mask);
    Reg::Duplicate(infReg, (float &)F32_INF, mask);
    Reg::Muls(infReg, infReg, infSign, mask_zero);
    Reg::Select(dstReg, infReg, dstReg, mask_zero);

    // x == -∞
    Reg::Muls(negInfReg, posInfReg, static_cast<T>(-1.0), mask);
    Reg::Compare<T, CMPMODE::EQ>(mask_neg_inf, srcReg, negInfReg, mask);
    if (negInfIsNaN) {
      Reg::Duplicate(infReg, (float &)F32_NAN, mask);
    } else {
      Reg::Duplicate(infReg, (float &)F32_INF, mask);
      Reg::Muls(infReg, infReg, infSign, mask_neg_inf);
    }
    Reg::Select(dstReg, infReg, dstReg, mask_neg_inf);
    Reg::StoreAlign(dst + i * vlSize, dstReg, mask);
  }
}

}  // namespace PolyGammaAPI

// ============================================================================
// PolyGamma: __aicore__ entry point for the polygamma function
//
// Computes the n-th derivative of the log-Gamma function:
//   psi^{(n)}(x) = d^{n+1}/dx^{n+1} ln(Gamma(x))
//
// Dispatch:
//   n = 0  → Digamma intrinsic
//   n = 1  → TriGammaImplVF
//   n >= 2 → (-1)^{n+1} * exp(lgamma(n+1)) * Zeta(n+1, x)
// ============================================================================
template <typename T, bool isReuseSource = false>
__aicore__ inline void PolyGammaExtend(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                       const int32_t n, const LocalTensor<T> &sharedTmpBuffer,
                                       const uint32_t calCount) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(SupportType<T, float>(), "PolyGamma only supports float on current device!");

  if (calCount == 0) {
    return;
  }

  ASCENDC_ASSERT((n >= 0),
                 { KERNEL_LOG(KERNEL_ERROR, "PolyGamma only supports non-negative integer order n, got n=%d", n); });

  if (n == 0) {
    // n=0: Digamma
    Digamma(dstTensor, srcTensor, calCount);
  } else if (n == 1) {
    PolyGammaAPI::TriGammaImplVF<T>((__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(),
                                    calCount);
    PipeBarrier<PIPE_V>();
    PolyGammaAPI::PolyGammaEdgeFixupVF<T>((__ubuf__ T *)dstTensor.GetPhyAddr(), (__ubuf__ T *)srcTensor.GetPhyAddr(),
                                          static_cast<T>(1.0), true, calCount);
  } else {
    // n>=2: psi^{(n)}(x) = (-1)^{n+1} * exp(lgamma(n+1)) * zeta(n+1, x)
    T nPlusOne = static_cast<T>(n + 1);
    T sign = (n % 2 == 0) ? static_cast<T>(-1.0) : static_cast<T>(1.0);

    __ubuf__ T *tmpUb = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr();
    PolyGammaAPI::BroadcastScalarImplVF<T>(tmpUb, nPlusOne, calCount);

    LocalTensor<uint8_t> zetaTmpBuf;
    ZetaExtend<T>(dstTensor, sharedTmpBuffer, srcTensor, zetaTmpBuf, calCount);
    PipeBarrier<PIPE_V>();

    // coeff = sign * n! = sign * exp(lgamma(n+1))
    Lgamma<T, true>(sharedTmpBuffer, sharedTmpBuffer, calCount);
    PipeBarrier<PIPE_V>();
    Exp(sharedTmpBuffer, sharedTmpBuffer, static_cast<int32_t>(calCount));
    PipeBarrier<PIPE_V>();

    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    PolyGammaAPI::PolyGammaMulVecVF<T>(dstUb, tmpUb, calCount);
    PipeBarrier<PIPE_V>();
    if (sign < static_cast<T>(0)) {
      PolyGammaAPI::PolyGammaMulCoeffVF<T>(dstUb, static_cast<T>(-1.0), calCount);
      PipeBarrier<PIPE_V>();
    }
    PolyGammaAPI::PolyGammaEdgeFixupVF<T>(dstUb, (__ubuf__ T *)srcTensor.GetPhyAddr(), sign, false, calCount);
  }
}

template <typename T, typename U>
__aicore__ inline void PolyGammaExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                       const AscendC::LocalTensor<U> &n,
                                       const AscendC::LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount) {
  static_assert(SupportType<U, int32_t>(), "PolyGamma order only supports int32_t");
  auto tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();
  PolyGammaExtend(dst, src, n.GetValue(0), tmpBuffer, calCount);
}

}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_POLYGAMMA_H__
