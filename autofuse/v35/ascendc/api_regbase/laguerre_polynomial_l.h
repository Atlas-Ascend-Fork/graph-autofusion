/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_LAGUERRE_POLYNOMIAL_L_H__
#define __ASCENDC_API_REGBASE_LAGUERRE_POLYNOMIAL_L_H__

static constexpr AscendC::Reg::DivSpecificMode kLaguerreHighPrecisionDivMode = {AscendC::Reg::MaskMergeMode::ZEROING,
                                                                                true};

template <typename T>
__simd_vf__ inline void LaguerrePolynomialLFillVF(__ubuf__ T *yAddr, uint32_t calCount, T fillValue) {
  const uint32_t repeatElem = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  const uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, repeatElem));

  AscendC::Reg::MaskReg activeMask;
  AscendC::Reg::RegTensor<T> yReg;
  AscendC::Reg::Duplicate(yReg, fillValue);
  for (uint16_t i = 0U; i < repeatTime; ++i) {
    activeMask = AscendC::Reg::UpdateMask<T>(calCount);
    AscendC::Reg::StoreAlign(yAddr + i * repeatElem, yReg, activeMask);
  }
}

template <typename T>
__simd_vf__ inline void LaguerrePolynomialLLinearVF(__ubuf__ T *xAddr, __ubuf__ T *yAddr, uint32_t calCount) {
  const uint32_t repeatElem = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  const uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, repeatElem));

  AscendC::Reg::MaskReg activeMask;
  AscendC::Reg::RegTensor<T> xReg;
  AscendC::Reg::RegTensor<T> oneReg;
  AscendC::Reg::RegTensor<T> outReg;
  AscendC::Reg::Duplicate(oneReg, static_cast<T>(1.0f));
  for (uint16_t i = 0U; i < repeatTime; ++i) {
    activeMask = AscendC::Reg::UpdateMask<T>(calCount);
    AscendC::Reg::LoadAlign(xReg, xAddr + i * repeatElem);
    AscendC::Reg::Sub(outReg, oneReg, xReg, activeMask);
    AscendC::Reg::StoreAlign(yAddr + i * repeatElem, outReg, activeMask);
  }
}

template <typename T>
__simd_vf__ inline void LaguerrePolynomialLImplVF(__ubuf__ T *xAddr, __ubuf__ T *yAddr, int32_t n, uint32_t calCount) {
  const uint32_t repeatElem = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  const uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calCount, repeatElem));

  AscendC::Reg::MaskReg activeMask;
  AscendC::Reg::MaskReg nonNanMask;
  AscendC::Reg::RegTensor<T> xReg;
  AscendC::Reg::RegTensor<T> oneReg;
  AscendC::Reg::RegTensor<T> twoReg;
  AscendC::Reg::RegTensor<T> kReg;
  AscendC::Reg::RegTensor<T> pReg;
  AscendC::Reg::RegTensor<T> pCompReg;
  AscendC::Reg::RegTensor<T> qReg;
  AscendC::Reg::RegTensor<T> qCompReg;
  AscendC::Reg::RegTensor<T> nextReg;
  AscendC::Reg::RegTensor<T> coefReg;
  AscendC::Reg::RegTensor<T> denReg;
  AscendC::Reg::RegTensor<T> aCompReg;
  AscendC::Reg::RegTensor<T> leftHiReg;
  AscendC::Reg::RegTensor<T> leftLoReg;
  AscendC::Reg::RegTensor<T> rightHiReg;
  AscendC::Reg::RegTensor<T> rightLoReg;
  AscendC::Reg::RegTensor<T> sumHiReg;
  AscendC::Reg::RegTensor<T> sumLoReg;
  AscendC::Reg::RegTensor<T> tmpReg0;
  AscendC::Reg::RegTensor<T> tmpReg1;
  AscendC::Reg::Duplicate(oneReg, static_cast<T>(1.0f));
  AscendC::Reg::Duplicate(twoReg, static_cast<T>(2.0f));

  for (uint16_t i = 0U; i < repeatTime; ++i) {
    activeMask = AscendC::Reg::UpdateMask<T>(calCount);
    AscendC::Reg::LoadAlign(xReg, xAddr + i * repeatElem);
    AscendC::Reg::Duplicate(pReg, static_cast<T>(1.0f), activeMask);
    AscendC::Reg::Duplicate(pCompReg, static_cast<T>(0.0f), activeMask);
    AscendC::Reg::Sub(qReg, oneReg, xReg, activeMask);
    AscendC::Reg::Sub(tmpReg0, oneReg, qReg, activeMask);
    AscendC::Reg::Sub(qCompReg, tmpReg0, xReg, activeMask);
    AscendC::Reg::Compare<T, AscendC::CMPMODE::EQ>(nonNanMask, xReg, xReg, activeMask);
    AscendC::Reg::Duplicate(kReg, static_cast<T>(1.0f), activeMask);
    AscendC::Reg::Duplicate(coefReg, static_cast<T>(3.0f), activeMask);
    AscendC::Reg::Duplicate(denReg, static_cast<T>(2.0f), activeMask);

    for (int32_t k = 0; k < n - 1; ++k) {
      // Keep each recurrence value as hi + lo to retain the cancellation residual.
      AscendC::Reg::Sub(nextReg, coefReg, xReg, nonNanMask);
      AscendC::Reg::Sub(tmpReg0, coefReg, nextReg, nonNanMask);
      AscendC::Reg::Sub(aCompReg, tmpReg0, xReg, nonNanMask);

      AscendC::Reg::Mul(leftHiReg, nextReg, qReg, nonNanMask);
      AscendC::Reg::Neg(tmpReg0, leftHiReg, nonNanMask);
      AscendC::Reg::Mul(leftLoReg, nextReg, oneReg, nonNanMask);
      AscendC::Reg::FusedMulDstAdd(leftLoReg, qReg, tmpReg0, nonNanMask);
      AscendC::Reg::Mul(tmpReg0, nextReg, qCompReg, nonNanMask);
      AscendC::Reg::Add(leftLoReg, leftLoReg, tmpReg0, nonNanMask);
      AscendC::Reg::Mul(tmpReg0, aCompReg, qReg, nonNanMask);
      AscendC::Reg::Add(leftLoReg, leftLoReg, tmpReg0, nonNanMask);
      AscendC::Reg::Mul(aCompReg, qCompReg, oneReg, nonNanMask);

      AscendC::Reg::Mul(rightHiReg, pReg, kReg, nonNanMask);
      AscendC::Reg::Neg(tmpReg0, rightHiReg, nonNanMask);
      AscendC::Reg::Mul(rightLoReg, pReg, oneReg, nonNanMask);
      AscendC::Reg::FusedMulDstAdd(rightLoReg, kReg, tmpReg0, nonNanMask);
      AscendC::Reg::Mul(tmpReg0, pCompReg, kReg, nonNanMask);
      AscendC::Reg::Add(rightLoReg, rightLoReg, tmpReg0, nonNanMask);

      AscendC::Reg::Sub(sumHiReg, leftHiReg, rightHiReg, nonNanMask);
      AscendC::Reg::Sub(tmpReg0, sumHiReg, leftHiReg, nonNanMask);
      AscendC::Reg::Sub(tmpReg1, sumHiReg, tmpReg0, nonNanMask);
      AscendC::Reg::Sub(sumLoReg, leftHiReg, tmpReg1, nonNanMask);
      AscendC::Reg::Neg(tmpReg1, rightHiReg, nonNanMask);
      AscendC::Reg::Sub(tmpReg1, tmpReg1, tmpReg0, nonNanMask);
      AscendC::Reg::Add(sumLoReg, sumLoReg, tmpReg1, nonNanMask);
      AscendC::Reg::Add(sumLoReg, sumLoReg, leftLoReg, nonNanMask);
      AscendC::Reg::Sub(sumLoReg, sumLoReg, rightLoReg, nonNanMask);

      AscendC::Reg::Div<T, &kLaguerreHighPrecisionDivMode>(nextReg, sumHiReg, denReg, nonNanMask);
      AscendC::Reg::Neg(tmpReg0, sumHiReg, nonNanMask);
      AscendC::Reg::Mul(tmpReg1, nextReg, oneReg, nonNanMask);
      AscendC::Reg::FusedMulDstAdd(tmpReg1, denReg, tmpReg0, nonNanMask);
      AscendC::Reg::Neg(tmpReg1, tmpReg1, nonNanMask);
      AscendC::Reg::Add(tmpReg1, tmpReg1, sumLoReg, nonNanMask);
      AscendC::Reg::Div<T, &kLaguerreHighPrecisionDivMode>(qCompReg, tmpReg1, denReg, nonNanMask);

      AscendC::Reg::Mul(pReg, qReg, oneReg, nonNanMask);
      AscendC::Reg::Mul(pCompReg, aCompReg, oneReg, nonNanMask);
      AscendC::Reg::Mul(qReg, nextReg, oneReg, nonNanMask);
      AscendC::Reg::Add(kReg, kReg, oneReg, activeMask);
      AscendC::Reg::Add(coefReg, coefReg, twoReg, activeMask);
      AscendC::Reg::Add(denReg, denReg, oneReg, activeMask);
    }

    AscendC::Reg::Add(qReg, qReg, qCompReg, nonNanMask);
    AscendC::Reg::StoreAlign(yAddr + i * repeatElem, qReg, activeMask);
  }
}

template <typename T, typename U>
__aicore__ inline void LaguerrePolynomialLExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                                 U n, const uint32_t calcount) {
  static_assert(AscendC::SupportType<T, float>(), "Current data type is not supported on current device!");
  const int32_t order = static_cast<int32_t>(n);

  __ubuf__ T *dstAddr = (__ubuf__ T *)dst.GetPhyAddr();

  if (order < 0) {
    LaguerrePolynomialLFillVF<T>(dstAddr, calcount, static_cast<T>(0.0f));
    return;
  }

  if (order == 0) {
    LaguerrePolynomialLFillVF<T>(dstAddr, calcount, static_cast<T>(1.0f));
    return;
  }

  __ubuf__ T *srcAddr = (__ubuf__ T *)src.GetPhyAddr();
  if (order == 1) {
    LaguerrePolynomialLLinearVF<T>(srcAddr, dstAddr, calcount);
    return;
  }

  LaguerrePolynomialLImplVF<T>(srcAddr, dstAddr, order, calcount);
}

template <typename T, typename U>
__aicore__ inline void LaguerrePolynomialLExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                                 const AscendC::LocalTensor<U> &n, const uint32_t calcount) {
  LaguerrePolynomialLExtend(dst, src, n.GetValue(0), calcount);
}

#endif  // __ASCENDC_API_REGBASE_LAGUERRE_POLYNOMIAL_L_H__
