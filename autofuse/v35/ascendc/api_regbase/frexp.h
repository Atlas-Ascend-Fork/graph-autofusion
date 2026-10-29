/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_FREXP_H__
#define __ASCENDC_API_REGBASE_FREXP_H__

using namespace AscendC;

template <typename T>
__simd_vf__ inline void FrexpImplVF(__ubuf__ uint32_t *mantissaUb, __ubuf__ int32_t *exponentUb,
                                    __ubuf__ uint32_t *srcUb, uint32_t calc_cnt) {
  uint32_t vl_size = static_cast<uint32_t>(GetVecLen() / sizeof(uint32_t));
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(calc_cnt, vl_size));

  Reg::RegTensor<uint32_t> srcReg, mantReg;
  Reg::RegTensor<uint32_t> expField, mantField, signField;
  Reg::MaskReg fullMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
  Reg::MaskReg zeroMask, infNanMask;

  Reg::RegTensor<uint32_t> mantMask, signMask, expMaskReg;
  Reg::RegTensor<uint32_t> frexpMantBiasReg, zeroReg, pow23Reg;

  Reg::Duplicate(mantMask, 0x007FFFFF, fullMask);
  Reg::Duplicate(signMask, 0x80000000, fullMask);
  Reg::Duplicate(expMaskReg, 0x7F800000, fullMask);
  Reg::Duplicate(frexpMantBiasReg, 0x3F000000, fullMask);
  Reg::Duplicate(zeroReg, 0, fullMask);
  Reg::Duplicate(pow23Reg, static_cast<uint32_t>(8388608), fullMask);

  Reg::MaskReg mask;
  Reg::MaskReg subnormWithZeroMask, tmpNotZeroMask, subnormOnlyMask, needsShiftMask;
  Reg::RegTensor<uint32_t> mantFieldSave, expFieldSave;

  for (uint16_t i = 0; i < repeat_time; ++i) {
    mask = Reg::UpdateMask<uint32_t>(calc_cnt);
    Reg::LoadAlign(srcReg, srcUb + i * vl_size);

    // Extract fields: mantissa, sign, exponent
    Reg::And(mantField, srcReg, mantMask, mask);
    Reg::And(signField, srcReg, signMask, mask);
    Reg::And(expField, srcReg, expMaskReg, mask);

    // Subnormal/zero detection before modifying expField
    Reg::Compare<uint32_t, CMPMODE::EQ>(subnormWithZeroMask, expField, zeroReg, mask);
    Reg::Compare<uint32_t, CMPMODE::EQ>(tmpNotZeroMask, mantField, zeroReg, mask);
    Reg::And(zeroMask, subnormWithZeroMask, tmpNotZeroMask, mask);
    Reg::Compare<uint32_t, CMPMODE::EQ>(infNanMask, expField, expMaskReg, mask);

    // Compute mantissa: (mant_field | sign_field | 0x3F000000)
    Reg::Or(mantReg, mantField, signField, mask);
    Reg::Or(mantReg, mantReg, frexpMantBiasReg, mask);

    // Compute exponent: E / 2^23 - 126
    Reg::Div(expField, expField, pow23Reg, mask);
    Reg::Adds((Reg::RegTensor<int32_t> &)(expField), (Reg::RegTensor<int32_t> &)(expField), static_cast<int16_t>(-126),
              mask);

    // Handle subnormals: normalize mantissa and correct exponent
    Reg::Not(tmpNotZeroMask, zeroMask, mask);
    Reg::And(subnormOnlyMask, subnormWithZeroMask, tmpNotZeroMask, mask);

    for (uint32_t si = 0; si < 23; ++si) {
      Reg::Compares<uint32_t, CMPMODE::LT>(needsShiftMask, mantField, static_cast<uint32_t>(1 << 22), subnormOnlyMask);
      Reg::Add(mantFieldSave, mantField, mantField, fullMask);
      Reg::Select(mantField, mantFieldSave, mantField, needsShiftMask);
      Reg::Adds(expFieldSave, expField, static_cast<int16_t>(-1), fullMask);
      Reg::Select(expField, expFieldSave, expField, needsShiftMask);
    }

    // Final normalization and mantReg assembly for subnormals
    Reg::Add(mantFieldSave, mantField, mantField, fullMask);
    Reg::And(mantFieldSave, mantFieldSave, mantMask, fullMask);
    Reg::Or(mantFieldSave, mantFieldSave, signField, fullMask);
    Reg::Or(mantFieldSave, mantFieldSave, frexpMantBiasReg, fullMask);
    Reg::Select(mantReg, mantFieldSave, mantReg, subnormOnlyMask);

    // Handle zero: mantissa preserves the input sign, exponent = 0
    Reg::Select(mantReg, signField, mantReg, zeroMask);
    Reg::Select((Reg::RegTensor<int32_t> &)(expField), (Reg::RegTensor<int32_t> &)(zeroReg),
                (Reg::RegTensor<int32_t> &)(expField), zeroMask);

    // Handle inf/nan: mantissa = src, exponent = 0
    Reg::Select(mantReg, srcReg, mantReg, infNanMask);
    Reg::Select((Reg::RegTensor<int32_t> &)(expField), (Reg::RegTensor<int32_t> &)(zeroReg),
                (Reg::RegTensor<int32_t> &)(expField), infNanMask);

    // Store results
    Reg::StoreAlign(mantissaUb + i * vl_size, mantReg, mask);
    Reg::StoreAlign(exponentUb + i * vl_size, (Reg::RegTensor<int32_t> &)(expField), mask);
  }
}

template <typename T>
__aicore__ inline void FrexpExtend(const LocalTensor<T> &mantissa, const LocalTensor<int32_t> &exponent,
                                   const LocalTensor<T> &src, const LocalTensor<uint8_t> &sharedTmpBuffer,
                                   const uint32_t calCount) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert((std::is_same_v<T, float>), "Frexp only supports float on current device!");

  LocalTensor<uint32_t> mantissaBits = mantissa.template ReinterpretCast<uint32_t>();
  LocalTensor<uint32_t> srcBits = src.template ReinterpretCast<uint32_t>();

  __ubuf__ uint32_t *mantissaUb = (__ubuf__ uint32_t *)mantissaBits.GetPhyAddr();
  __ubuf__ int32_t *exponentUb = (__ubuf__ int32_t *)exponent.GetPhyAddr();
  __ubuf__ uint32_t *srcUb = (__ubuf__ uint32_t *)srcBits.GetPhyAddr();
  FrexpImplVF<T>(mantissaUb, exponentUb, srcUb, calCount);
}

#endif  // __ASCENDC_API_REGBASE_FREXP_H__
