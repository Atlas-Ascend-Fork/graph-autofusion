/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ATAN2_H__
#define __ASCENDC_API_REGBASE_ATAN2_H__

using namespace AscendC;

constexpr float ATAN2_PI = 3.14159265358979323846264338327950288;
constexpr float ATAN2_HALF_PI = 1.57079632679489661923132169163975144;
constexpr float ATAN2_NEG_HALF_PI = 0 - ATAN2_HALF_PI;
constexpr float ATAN2_NEG_PI = 0 - ATAN2_PI;

/* atan2(y, x) 的实现：

1. 如果 x > 0：
   atan2(y, x) = atan(y/x)

2. 如果 x < 0：
   atan2(y, x) = atan(y/x) + π  (如果 y ≥ 0)
   atan2(y, x) = atan(y/x) - π  (如果 y < 0)

3. 如果 x == 0：
   atan2(y, x) = π/2   (如果 y > 0)
   atan2(y, x) = -π/2  (如果 y < 0)
   atan2(+0, +0) = +0, atan2(-0, +0) = +π
   atan2(+0, -0) = -0, atan2(-0, -0) = -π
*/

template <typename T>
__aicore__ inline void Atan2PrepareRatio(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint32_t calc_cnt) {
  constexpr uint32_t vl_size = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(calc_cnt, vl_size));
  using BitT = std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>;
  constexpr BitT kSignClearMask =
      sizeof(T) == sizeof(uint16_t) ? static_cast<BitT>(0x7fffU) : static_cast<BitT>(0x7fffffffU);
  constexpr BitT kMinNormalBits =
      sizeof(T) == sizeof(uint16_t) ? static_cast<BitT>(0x0400U) : static_cast<BitT>(0x00800000U);
  constexpr T kSubnormalScale = sizeof(T) == sizeof(uint16_t) ? static_cast<T>(1024.0f) : static_cast<T>(16777216.0f);

  MicroAPI::RegTensor<T> x_reg, y_reg, ratio_reg, scaled_x_reg, scaled_y_reg, scaled_ratio_reg;
  MicroAPI::RegTensor<BitT> x_abs_bits, y_abs_bits, clear_sign_mask_reg, min_normal_bits_reg;
  MicroAPI::MaskReg mask, x_subnormal_mask, y_subnormal_mask, x_nonzero_mask, y_nonzero_mask, subnormal_pair_mask;
  MicroAPI::MaskReg full_mask = MicroAPI::CreateMask<uint8_t>();
  MicroAPI::Duplicate(clear_sign_mask_reg, kSignClearMask, full_mask);
  MicroAPI::Duplicate(min_normal_bits_reg, kMinNormalBits, full_mask);

  for (uint16_t i = 0U; i < repeat_time; ++i) {
    mask = MicroAPI::UpdateMask<T>(calc_cnt);
    MicroAPI::LoadAlign(x_reg, src0 + i * vl_size);
    MicroAPI::LoadAlign(y_reg, src1 + i * vl_size);
    MicroAPI::Div(ratio_reg, y_reg, x_reg, mask);

    // Scale two nonzero subnormals together before division. Their ratio is
    // invariant, while the scaled operands are normal IEEE-754 values.
    MicroAPI::And(x_abs_bits, (MicroAPI::RegTensor<BitT> &)x_reg, clear_sign_mask_reg, mask);
    MicroAPI::And(y_abs_bits, (MicroAPI::RegTensor<BitT> &)y_reg, clear_sign_mask_reg, mask);
    MicroAPI::Compares<BitT, CMPMODE::LT>(x_subnormal_mask, x_abs_bits, kMinNormalBits, mask);
    MicroAPI::Compares<BitT, CMPMODE::LT>(y_subnormal_mask, y_abs_bits, kMinNormalBits, mask);
    MicroAPI::Compares<BitT, CMPMODE::NE>(x_nonzero_mask, x_abs_bits, static_cast<BitT>(0), mask);
    MicroAPI::Compares<BitT, CMPMODE::NE>(y_nonzero_mask, y_abs_bits, static_cast<BitT>(0), mask);
    MicroAPI::MaskAnd(x_subnormal_mask, x_subnormal_mask, x_nonzero_mask, mask);
    MicroAPI::MaskAnd(y_subnormal_mask, y_subnormal_mask, y_nonzero_mask, mask);
    MicroAPI::MaskAnd(subnormal_pair_mask, x_subnormal_mask, y_subnormal_mask, mask);
    MicroAPI::Muls(scaled_x_reg, x_reg, kSubnormalScale, subnormal_pair_mask);
    MicroAPI::Muls(scaled_y_reg, y_reg, kSubnormalScale, subnormal_pair_mask);
    MicroAPI::Div(scaled_ratio_reg, scaled_y_reg, scaled_x_reg, subnormal_pair_mask);
    MicroAPI::Select(ratio_reg, scaled_ratio_reg, ratio_reg, subnormal_pair_mask);
    MicroAPI::StoreAlign(dst + i * vl_size, ratio_reg, mask);
  }
}

template <typename T>
__aicore__ inline void AtanPostProcess(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ T *src_atan,
                                       uint32_t calc_cnt) {
  uint32_t vl_size = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(calc_cnt, vl_size));

  using BitT = std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>;
  MicroAPI::RegTensor<T> src0_reg, src1_reg, atan_reg, half_PI_reg, neg_half_PI_reg, PI_reg, neg_PI_reg;
  MicroAPI::RegTensor<T> dst_reg, pi_by_y_sign_reg, zero_case_reg;
  MicroAPI::RegTensor<BitT> sign_mask_reg, x_sign_reg, y_sign_reg;
  MicroAPI::MaskReg full_mask, dst_mask, x_zero_mask, y_zero_mask, x_negative_mask;
  MicroAPI::MaskReg x_negative_sign_mask, y_negative_sign_mask, x_negative_y_zero_mask;
  full_mask = MicroAPI::CreateMask<uint8_t>();
  if constexpr (sizeof(T) == sizeof(uint16_t)) {
    MicroAPI::Duplicate(sign_mask_reg, static_cast<BitT>(0x8000U), full_mask);
  } else {
    MicroAPI::Duplicate(sign_mask_reg, static_cast<BitT>(0x80000000U), full_mask);
  }
  MicroAPI::Duplicate(half_PI_reg, ATAN2_HALF_PI, full_mask);
  MicroAPI::Duplicate(neg_half_PI_reg, ATAN2_NEG_HALF_PI, full_mask);
  MicroAPI::Duplicate(PI_reg, ATAN2_PI, full_mask);
  MicroAPI::Duplicate(neg_PI_reg, ATAN2_NEG_PI, full_mask);

  MicroAPI::MaskReg mask;
  // mainBlock
  for (uint16_t i = 0U; i < repeat_time; ++i) {
    mask = AscendC::MicroAPI::UpdateMask<T>(calc_cnt);
    MicroAPI::LoadAlign(src0_reg, src0 + i * vl_size);
    MicroAPI::LoadAlign(src1_reg, src1 + i * vl_size);
    MicroAPI::LoadAlign(atan_reg, src_atan + i * vl_size);

    // 按照 y >= 0 或 y < 0 填充 PI 寄存器。
    MicroAPI::Compares<T, CMPMODE::GE>(dst_mask, src1_reg, (T)0.0, mask);
    MicroAPI::Select(dst_reg, PI_reg, neg_PI_reg, dst_mask);
    // atan结果与PI寄存器相加
    MicroAPI::Add(dst_reg, dst_reg, atan_reg, mask);

    // x > 0时，atan2(y, x) = atan(y/x)，否则与PI寄存器相加
    MicroAPI::Compares<T, CMPMODE::GT>(dst_mask, src0_reg, (T)0.0, mask);
    MicroAPI::Select(dst_reg, atan_reg, dst_reg, dst_mask);

    // Preserve IEEE-754 signed-zero behavior. Numerical comparisons alone
    // cannot distinguish +0 from -0, so inspect the sign bits explicitly.
    MicroAPI::And(x_sign_reg, (MicroAPI::RegTensor<BitT> &)src0_reg, sign_mask_reg, mask);
    MicroAPI::And(y_sign_reg, (MicroAPI::RegTensor<BitT> &)src1_reg, sign_mask_reg, mask);
    MicroAPI::Compares<BitT, CMPMODE::NE>(x_negative_sign_mask, x_sign_reg, static_cast<BitT>(0), mask);
    MicroAPI::Compares<BitT, CMPMODE::NE>(y_negative_sign_mask, y_sign_reg, static_cast<BitT>(0), mask);
    MicroAPI::Compares<T, CMPMODE::EQ>(x_zero_mask, src0_reg, (T)0.0, mask);
    MicroAPI::Compares<T, CMPMODE::EQ>(y_zero_mask, src1_reg, (T)0.0, mask);

    // For x < 0 and y = +/-0, use y's sign to select +/-pi.
    MicroAPI::Select(pi_by_y_sign_reg, neg_PI_reg, PI_reg, y_negative_sign_mask);
    MicroAPI::Compares<T, CMPMODE::LT>(x_negative_mask, src0_reg, (T)0.0, mask);
    MicroAPI::MaskAnd(x_negative_y_zero_mask, x_negative_mask, y_zero_mask, mask);
    MicroAPI::Select(dst_reg, pi_by_y_sign_reg, dst_reg, x_negative_y_zero_mask);

    // Handle x == 0. y > 0 maps to +pi/2 and y < 0 maps to -pi/2.
    MicroAPI::Compares<T, CMPMODE::GT>(dst_mask, src1_reg, (T)0.0, mask);
    MicroAPI::Select(atan_reg, half_PI_reg, atan_reg, dst_mask);

    // y < 0 时，atan2(y, x) = -pi/2。
    MicroAPI::Compares<T, CMPMODE::LT>(dst_mask, src1_reg, (T)0.0, mask);
    MicroAPI::Select(atan_reg, neg_half_PI_reg, atan_reg, dst_mask);

    // For x = +/-0 and y = +/-0, select signed y-zero when x is +0;
    // select signed pi when x is -0.
    MicroAPI::Select(zero_case_reg, pi_by_y_sign_reg, src1_reg, x_negative_sign_mask);
    MicroAPI::Select(atan_reg, zero_case_reg, atan_reg, y_zero_mask);
    MicroAPI::Select(dst_reg, atan_reg, dst_reg, x_zero_mask);

    MicroAPI::StoreAlign(dst + i * vl_size, dst_reg, mask);
  }
}

template <typename T>
__aicore__ inline void Atan2Extend(const LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<T> &src1,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calc_cnt) {
  VF_CALL<Atan2PrepareRatio<T>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src0.GetPhyAddr(),
                                (__ubuf__ T *)src1.GetPhyAddr(), calc_cnt);
  AscendC::Atan(dst, dst, tmp_buf, calc_cnt);
  VF_CALL<AtanPostProcess<T>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src0.GetPhyAddr(),
                              (__ubuf__ T *)src1.GetPhyAddr(), (__ubuf__ T *)dst.GetPhyAddr(), calc_cnt);
}
#endif
