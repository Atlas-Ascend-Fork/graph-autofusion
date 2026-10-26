/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAST_H
#define CAST_H
using namespace AscendC;

template <typename InT, typename OutT>
__simd_callee__ inline constexpr AscendC::RoundMode GetCastRoundMode() {
  bool getRint = SupportType<Tuple<OutT, InT>, Tuple<half, int16_t>, Tuple<bfloat16_t, float>>();
  bool getTrunc = SupportType<Tuple<OutT, InT>, Tuple<int32_t, bfloat16_t>, Tuple<float, int32_t>, Tuple<half, int32_t>,
                              Tuple<float, float>, Tuple<int4x2_t, half>>();

  if (getRint) {
    return AscendC::RoundMode::CAST_RINT;
  }
  if (getTrunc) {
    return AscendC::RoundMode::CAST_TRUNC;
  }
  if constexpr (AscendC::IsSameType<InT, float>::value) {
    if constexpr (AscendC::IsSameType<OutT, half>::value) {
      return AscendC::RoundMode::CAST_RINT;
    }
    if constexpr (AscendC::SupportType<OutT, int64_t, int32_t, int16_t>()) {
      return AscendC::RoundMode::CAST_TRUNC;
    }
  }
  if constexpr (AscendC::IsSameType<InT, half>::value &&
                AscendC::SupportType<OutT, int32_t, int16_t, int8_t, uint8_t>()) {
    return AscendC::RoundMode::CAST_TRUNC;
  }
  if constexpr (AscendC::IsSameType<InT, int64_t>::value && AscendC::SupportType<OutT, float>()) {
    return AscendC::RoundMode::CAST_RINT;
  }
  return AscendC::RoundMode::CAST_RINT;
}

template <typename InT, typename OutT>
__simd_callee__ inline void GenLoadInstr(AscendC::Reg::RegTensor<InT> &srcVreg, __ubuf__ InT *srcAddr) {
  if constexpr (SupportType<InT, int4x2_t>() && sizeof(OutT) == 2) {
    AscendC::Reg::UnPack<uint16_t, uint8_t>((AscendC::Reg::RegTensor<uint16_t> &)srcVreg,
                                            (AscendC::Reg::RegTensor<uint8_t> &)srcVreg);
    AscendC::Reg::UnPack<uint32_t, uint16_t>((AscendC::Reg::RegTensor<uint32_t> &)srcVreg,
                                             (AscendC::Reg::RegTensor<uint16_t> &)srcVreg);
  } else if constexpr (sizeof(InT) == 1 && sizeof(OutT) == 2) {
    AscendC::Reg::DataCopy<InT, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(srcVreg, srcAddr);
  } else if constexpr (sizeof(InT) == 2 && sizeof(OutT) == 4) {
    AscendC::Reg::DataCopy<InT, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcVreg, srcAddr);
  } else if constexpr (sizeof(InT) == 1 && sizeof(OutT) == 4) {
    AscendC::Reg::DataCopy<InT, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(srcVreg, srcAddr);
  } else {
    AscendC::Reg::DataCopy(srcVreg, srcAddr);
  }
}

template <typename InT, typename OutT>
__simd_callee__ inline void GenStoreInstr(__ubuf__ OutT *dstAddr, AscendC::Reg::RegTensor<OutT> &dstVreg,
                                          AscendC::Reg::MaskReg &maskReg) {
  if constexpr (SupportType<OutT, int4x2_t>() && sizeof(InT) == 2) {
    AscendC::Reg::Pack<uint16_t, uint32_t>((AscendC::Reg::RegTensor<uint16_t> &)dstVreg,
                                           (AscendC::Reg::RegTensor<uint32_t> &)dstVreg);
    AscendC::Reg::Pack<uint8_t, uint16_t>((AscendC::Reg::RegTensor<uint8_t> &)dstVreg,
                                          (AscendC::Reg::RegTensor<uint16_t> &)dstVreg);
  } else if constexpr (sizeof(OutT) == 1 && sizeof(InT) == 2) {
    AscendC::Reg::DataCopy<OutT, AscendC::Reg::StoreDist::DIST_PACK_B16>(dstAddr, dstVreg, maskReg);
  } else if constexpr (sizeof(OutT) == 2 && sizeof(InT) == 4) {
    AscendC::Reg::DataCopy<OutT, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstAddr, dstVreg, maskReg);
  } else if constexpr (sizeof(OutT) == 1 && sizeof(InT) == 4) {
    AscendC::Reg::DataCopy<OutT, AscendC::Reg::StoreDist::DIST_PACK4_B32>(dstAddr, dstVreg, maskReg);
  } else {
    AscendC::Reg::DataCopy(dstAddr, dstVreg, maskReg);
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendCommon(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                         uint32_t repeatTimes, uint32_t innerLoopStride) {
  static constexpr AscendC::Reg::CastTrait castTrait0 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                         AscendC::Reg::MaskMergeMode::ZEROING, roundMode};
  static constexpr AscendC::Reg::CastTrait castTrait1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::UNKNOWN};

  uint32_t sreg = static_cast<uint32_t>(count);
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  AscendC::Reg::MaskReg stMaskReg, exMaskReg;
  exMaskReg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

  for (uint32_t i = 0; i < repeatTimes; i++) {
    if constexpr (sizeof(InT) < sizeof(OutT)) {
      stMaskReg = AscendC::Reg::UpdateMask<OutT>(sreg);
    } else {
      stMaskReg = AscendC::Reg::UpdateMask<InT>(sreg);
    }
    if constexpr ((SupportType<OutT, int4x2_t>() && sizeof(InT) == 2)) {
      AscendC::Reg::MaskPack(stMaskReg, stMaskReg);
    }
    GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i);
    if constexpr (std::is_same_v<InT, int32_t> && std::is_same_v<OutT, half>) {
      AscendC::Reg::Cast<float, InT, castTrait0>((AscendC::Reg::RegTensor<float> &)dstVreg, srcVreg, exMaskReg);
      AscendC::Reg::Cast<OutT, float, castTrait0>(dstVreg, (AscendC::Reg::RegTensor<float> &)dstVreg, exMaskReg);
    } else if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
      AscendC::Reg::Truncate<InT, roundMode>(dstVreg, srcVreg, exMaskReg);
    } else if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, int32_t>) {
      // CAST_TRUNC does not saturate finite float32 overflow. Apply deterministic saturation explicitly.
      AscendC::Reg::MaskReg upperOverflowMask, lowerOverflowMask, nanMask;
      AscendC::Reg::RegTensor<int32_t> upperLimitReg, lowerLimitReg, zeroReg;
      AscendC::Reg::Compares<float, AscendC::CMPMODE::GE>(upperOverflowMask, srcVreg, 2147483648.0f, exMaskReg);
      AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(lowerOverflowMask, srcVreg, -2147483648.0f, exMaskReg);
      AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nanMask, srcVreg, srcVreg, exMaskReg);
      AscendC::Reg::Cast<OutT, InT, castTrait0>(dstVreg, srcVreg, exMaskReg);
      AscendC::Reg::Duplicate(upperLimitReg, static_cast<int32_t>(2147483647), exMaskReg);
      AscendC::Reg::Duplicate(lowerLimitReg, static_cast<int32_t>(-2147483647 - 1), exMaskReg);
      AscendC::Reg::Duplicate(zeroReg, static_cast<int32_t>(0), exMaskReg);
      AscendC::Reg::Select(dstVreg, upperLimitReg, dstVreg, upperOverflowMask);
      AscendC::Reg::Select(dstVreg, lowerLimitReg, dstVreg, lowerOverflowMask);
      AscendC::Reg::Select(dstVreg, zeroReg, dstVreg, nanMask);
    } else if constexpr (std::is_same_v<InT, bfloat16_t> && std::is_same_v<OutT, float>) {
      AscendC::Reg::Cast<OutT, InT, castTrait1>(dstVreg, srcVreg, exMaskReg);
    } else {
      AscendC::Reg::Cast<OutT, InT, castTrait0>(dstVreg, srcVreg, exMaskReg);
    }
    GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendB4(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                     uint32_t repeatTimes, uint32_t innerLoopStride) {
  static constexpr AscendC::Reg::CastTrait cast_trait = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                         AscendC::Reg::MaskMergeMode::ZEROING, roundMode};

  AscendC::Reg::MaskReg ldMaskReg, stMaskReg, exMaskReg, dumpMaskReg;
  uint32_t sreg = static_cast<uint32_t>(count);
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  exMaskReg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

  for (uint32_t i = 0; i < repeatTimes; i++) {
    if constexpr (sizeof(InT) < sizeof(OutT)) {
      stMaskReg = AscendC::Reg::UpdateMask<OutT>(sreg);
      exMaskReg = stMaskReg;
      AscendC::Reg::MaskPack(ldMaskReg, stMaskReg);
      if constexpr ((SupportType<OutT, int4x2_t>() && sizeof(InT) == 2)) {
        AscendC::Reg::MaskPack(ldMaskReg, ldMaskReg);
        AscendC::Reg::MaskUnPack(stMaskReg, ldMaskReg);
        AscendC::Reg::MaskUnPack(exMaskReg, stMaskReg);
        AscendC::Reg::MaskInterleave<uint16_t>(stMaskReg, dumpMaskReg, stMaskReg, stMaskReg);
      }
    } else if constexpr (sizeof(InT) > sizeof(OutT)) {
      ldMaskReg = AscendC::Reg::UpdateMask<InT>(sreg);
      exMaskReg = ldMaskReg;
      AscendC::Reg::MaskPack(stMaskReg, ldMaskReg);
      if constexpr ((SupportType<OutT, int4x2_t>() && sizeof(InT) == 2)) {
        AscendC::Reg::MaskPack(stMaskReg, stMaskReg);
      }
    }
    if constexpr (std::is_same_v<InT, int4x2_t>) {
      GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i / 2);
    } else {
      GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i);
    }
    AscendC::Reg::Cast<OutT, InT, cast_trait>(dstVreg, srcVreg, exMaskReg);
    if constexpr (std::is_same_v<OutT, int4x2_t>) {
      GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i / 2, dstVreg, stMaskReg);
    } else {
      GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
    }
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendB8(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                     uint32_t repeatTimes, uint32_t innerLoopStride) {
  static constexpr AscendC::Reg::CastTrait cast_trait_in = {
      AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
      GetCastRoundMode<InT, half>()};
  static constexpr AscendC::Reg::CastTrait cast_trait_out = {
      AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
      GetCastRoundMode<half, OutT>()};

  uint32_t sreg = static_cast<uint32_t>(count);
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  AscendC::Reg::RegTensor<half> tmpVreg;
  AscendC::Reg::MaskReg ldMaskReg, stMaskReg, exMaskReg, dumpMaskReg;
  exMaskReg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

  if constexpr (sizeof(OutT) == sizeof(float)) {
    sreg = sreg * 2;
  }

  for (uint32_t i = 0; i < repeatTimes; i++) {
    stMaskReg = AscendC::Reg::UpdateMask<half>(sreg);
    GenLoadInstr<InT, half>(srcVreg, srcUb + innerLoopStride * i);
    AscendC::Reg::Cast<half, InT, cast_trait_in>(tmpVreg, srcVreg, exMaskReg);
    if constexpr (sizeof(half) < sizeof(OutT)) {
      AscendC::Reg::UnPack<uint32_t, uint16_t>((AscendC::Reg::RegTensor<uint32_t> &)tmpVreg,
                                               (AscendC::Reg::RegTensor<uint16_t> &)tmpVreg);
    }
    AscendC::Reg::Cast<OutT, half, cast_trait_out>(dstVreg, tmpVreg, exMaskReg);
    if constexpr (SupportType<OutT, int4x2_t>()) {
      GenStoreInstr<half, OutT>(dstUb + innerLoopStride * i / 2, dstVreg, stMaskReg);
    } else {
      GenStoreInstr<half, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
    }
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendInt8ToFloat(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                              uint32_t repeatTimes, uint32_t innerLoopStride) {
  static constexpr AscendC::Reg::CastTrait castTrait = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::UNKNOWN};

  uint32_t sreg = static_cast<uint32_t>(count);
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  AscendC::Reg::RegTensor<half> tmpVreg;
  AscendC::Reg::MaskReg stMaskReg, exMaskReg;
  exMaskReg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

  for (uint32_t i = 0; i < repeatTimes; i++) {
    stMaskReg = AscendC::Reg::UpdateMask<OutT>(sreg);
    GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i);
    AscendC::Reg::Cast<half, InT, castTrait>(tmpVreg, srcVreg, exMaskReg);
    AscendC::Reg::Cast<OutT, half, castTrait>(dstVreg, tmpVreg, exMaskReg);
    GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendFloatToInt8(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                              uint32_t repeatTimes, uint32_t innerLoopStride) {
  static constexpr AscendC::Reg::CastTrait castTraitFloatToHalf = {
      AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
      AscendC::RoundMode::CAST_TRUNC};

  static constexpr AscendC::Reg::CastTrait castTraitHalfToInt8 = {
      AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
      AscendC::RoundMode::CAST_TRUNC};

  uint32_t sreg = static_cast<uint32_t>(count);
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  AscendC::Reg::RegTensor<half> tmpVreg;
  AscendC::Reg::MaskReg stMaskReg, exMaskReg;
  exMaskReg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

  for (uint32_t i = 0; i < repeatTimes; i++) {
    stMaskReg = AscendC::Reg::UpdateMask<InT>(sreg);
    GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i);
    AscendC::Reg::Cast<half, InT, castTraitFloatToHalf>(tmpVreg, srcVreg, exMaskReg);
    AscendC::Reg::Cast<OutT, half, castTraitHalfToInt8>(dstVreg, tmpVreg, exMaskReg);
    GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendInt8ToInt16(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                              uint32_t repeatTimes, uint32_t innerLoopStride) {
  static constexpr AscendC::Reg::CastTrait castTrait = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::UNKNOWN};

  uint32_t sreg = static_cast<uint32_t>(count);
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  AscendC::Reg::MaskReg stMaskReg, exMaskReg;
  exMaskReg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

  for (uint32_t i = 0; i < repeatTimes; i++) {
    stMaskReg = AscendC::Reg::UpdateMask<OutT>(sreg);
    GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i);
    AscendC::Reg::Cast<OutT, InT, castTrait>(dstVreg, srcVreg, exMaskReg);
    GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendInt16ToInt8(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                              uint32_t repeatTimes, uint32_t innerLoopStride) {
  static constexpr AscendC::Reg::CastTrait castTraitHalfToInt8 = {
      AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
      AscendC::RoundMode::CAST_TRUNC};

  static constexpr AscendC::Reg::CastTrait castTraitInt16ToHalf = {
      AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
      AscendC::RoundMode::CAST_RINT};

  uint32_t sreg = static_cast<uint32_t>(count);
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  AscendC::Reg::RegTensor<half> tmpVreg;
  AscendC::Reg::MaskReg stMaskReg, exMaskReg;
  exMaskReg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

  for (uint32_t i = 0; i < repeatTimes; i++) {
    stMaskReg = AscendC::Reg::UpdateMask<InT>(sreg);
    GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i);
    AscendC::Reg::Cast<half, InT, castTraitInt16ToHalf>(tmpVreg, srcVreg, exMaskReg);
    AscendC::Reg::Cast<OutT, half, castTraitHalfToInt8>(dstVreg, tmpVreg, exMaskReg);
    GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendInt16ToUint8(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                               uint32_t repeatTimes, uint32_t innerLoopStride) {
  static constexpr AscendC::Reg::CastTrait castTrait = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::UNKNOWN};

  uint32_t sreg = static_cast<uint32_t>(count);
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  AscendC::Reg::MaskReg stMaskReg, exMaskReg;
  exMaskReg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

  for (uint32_t i = 0; i < repeatTimes; i++) {
    stMaskReg = AscendC::Reg::UpdateMask<InT>(sreg);
    GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i);
    AscendC::Reg::Cast<OutT, InT, castTrait>(dstVreg, srcVreg, exMaskReg);
    GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendFloatBool(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                            uint32_t repeatTimes, uint32_t innerLoopStride) {
  uint32_t sreg = static_cast<uint32_t>(count);
  int16_t shift = 31;
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<uint32_t> negVreg, u32Vreg, unsignVreg, notVreg;
  AscendC::Reg::MaskReg stMaskReg, exMaskReg;
  exMaskReg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
  AscendC::Reg::Duplicate(unsignVreg, 0x7FFFFFFF, exMaskReg);

  for (uint32_t i = 0; i < repeatTimes; i++) {
    stMaskReg = AscendC::Reg::UpdateMask<float>(sreg);
    AscendC::Reg::DataCopy(srcVreg, srcUb + innerLoopStride * i);
    AscendC::Reg::And(u32Vreg, (AscendC::Reg::RegTensor<uint32_t> &)srcVreg, unsignVreg, exMaskReg);
    AscendC::Reg::Not(notVreg, u32Vreg, exMaskReg);
    AscendC::Reg::Adds(negVreg, notVreg, 1u, exMaskReg);
    AscendC::Reg::Or(u32Vreg, negVreg, u32Vreg, exMaskReg);
    AscendC::Reg::ShiftRights(u32Vreg, u32Vreg, shift, exMaskReg);
    AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
        (__ubuf__ uint8_t *)(dstUb + innerLoopStride * i), (AscendC::Reg::RegTensor<uint8_t> &)u32Vreg, stMaskReg);
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendInt64Bool(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                            uint32_t repeatTimes, uint32_t innerLoopStride) {
  AscendC::Reg::RegTensor<int64_t> oneReg, zeroReg;
  AscendC::Reg::MaskReg b64MaskReg, b32MaskReg, allMaskReg, cmpMaskReg;
  AscendC::Reg::RegTensor<InT> srcReg, tmpReg;
  AscendC::Reg::RegTensor<uint32_t> dstReg;
  allMaskReg = AscendC::Reg::CreateMask<int64_t, AscendC::Reg::MaskPattern::ALL>();
  AscendC::Reg::Duplicate(oneReg, 0x1, allMaskReg);
  AscendC::Reg::Duplicate(zeroReg, 0x0, allMaskReg);

  for (uint32_t i = 0; i < repeatTimes; i++) {
    uint32_t blockCount = static_cast<uint32_t>(count) - i * innerLoopStride;
    blockCount = blockCount < innerLoopStride ? blockCount : innerLoopStride;
    uint32_t b64Sreg = blockCount;
    uint32_t b32Sreg = blockCount;
    b64MaskReg = AscendC::Reg::UpdateMask<int64_t>(b64Sreg);
    b32MaskReg = AscendC::Reg::UpdateMask<int32_t>(b32Sreg);
    AscendC::Reg::DataCopy(srcReg, srcUb + innerLoopStride * i);
    AscendC::Reg::Compares<InT, AscendC::CMPMODE::EQ>(cmpMaskReg, srcReg, static_cast<InT>(0), b64MaskReg);
    AscendC::Reg::Select(tmpReg, zeroReg, oneReg, cmpMaskReg);
    AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t> &)dstReg, (AscendC::Reg::RegTensor<InT> &)tmpReg);
    AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
        (__ubuf__ uint8_t *)(dstUb + innerLoopStride * i), (AscendC::Reg::RegTensor<uint8_t> &)dstReg, b32MaskReg);
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendInt32Bool(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                            uint32_t repeatTimes, uint32_t innerLoopStride) {
  uint32_t sreg = static_cast<uint32_t>(count);
  AscendC::Reg::RegTensor<int32_t> oneReg, zeroReg;
  AscendC::Reg::MaskReg stMaskReg, allMaskReg, cmpMaskReg;
  AscendC::Reg::RegTensor<InT> srcReg;
  AscendC::Reg::RegTensor<InT> dstReg;
  allMaskReg = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
  AscendC::Reg::Duplicate(oneReg, 0x1, allMaskReg);
  AscendC::Reg::Duplicate(zeroReg, 0x0, allMaskReg);

  for (uint32_t i = 0; i < repeatTimes; i++) {
    stMaskReg = AscendC::Reg::UpdateMask<int32_t>(sreg);
    AscendC::Reg::DataCopy(srcReg, srcUb + innerLoopStride * i);
    AscendC::Reg::Compares<InT, AscendC::CMPMODE::EQ>(cmpMaskReg, srcReg, static_cast<InT>(0), stMaskReg);
    AscendC::Reg::Select(dstReg, zeroReg, oneReg, cmpMaskReg);
    AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
        (__ubuf__ uint8_t *)(dstUb + innerLoopStride * i), (AscendC::Reg::RegTensor<uint8_t> &)dstReg, stMaskReg);
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendSmallIntBool(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                               uint32_t repeatTimes, uint32_t innerLoopStride) {
  uint32_t sreg = static_cast<uint32_t>(count);
  AscendC::Reg::RegTensor<InT> oneReg, zeroReg, srcReg, dstReg;
  AscendC::Reg::MaskReg stMaskReg, allMaskReg, cmpMaskReg;
  allMaskReg = AscendC::Reg::CreateMask<InT, AscendC::Reg::MaskPattern::ALL>();
  AscendC::Reg::Duplicate(oneReg, 0x1, allMaskReg);
  AscendC::Reg::Duplicate(zeroReg, 0x0, allMaskReg);

  for (uint32_t i = 0; i < repeatTimes; i++) {
    stMaskReg = AscendC::Reg::UpdateMask<InT>(sreg);
    AscendC::Reg::DataCopy(srcReg, srcUb + innerLoopStride * i);
    AscendC::Reg::Compares<InT, AscendC::CMPMODE::EQ>(cmpMaskReg, srcReg, static_cast<InT>(0), stMaskReg);
    AscendC::Reg::Select(dstReg, zeroReg, oneReg, cmpMaskReg);
    if constexpr (sizeof(InT) == 2) {
      AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::StoreDist::DIST_PACK_B16>(
          (__ubuf__ uint8_t *)(dstUb + innerLoopStride * i), (AscendC::Reg::RegTensor<uint8_t> &)dstReg, stMaskReg);
    } else {
      AscendC::Reg::DataCopy((__ubuf__ uint8_t *)(dstUb + innerLoopStride * i),
                             (AscendC::Reg::RegTensor<uint8_t> &)dstReg, stMaskReg);
    }
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendB64(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                      uint32_t repeatTimes, uint32_t innerLoopStride) {
  static constexpr AscendC::Reg::CastTrait cast_trait_b64 = {
      AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
      AscendC::RoundMode::CAST_TRUNC};

  uint32_t b64_sreg = GetVecLen() / sizeof(int64_t);
  uint32_t b32_sreg = GetVecLen() / sizeof(int64_t);
  uint32_t tailnum = count % innerLoopStride;
  uint32_t tailCtrl = (tailnum >= 0) ? 1 : 0;
  tailnum = (tailnum == 0) ? innerLoopStride : tailnum;

  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg, tmpoutVreg;
  AscendC::Reg::MaskReg maskReg, b64MaskReg, b32MaskReg, tailMaskReg;
  AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

  b64MaskReg = AscendC::Reg::UpdateMask<int64_t>(b64_sreg);
  b32MaskReg = AscendC::Reg::UpdateMask<int32_t>(b32_sreg);

  AscendC::Reg::MaskReg fullMask2 = AscendC::Reg::CreateMask<uint64_t, AscendC::Reg::MaskPattern::ALL>();

  for (uint32_t i = 0; i < (repeatTimes - 1); i++) {
    if constexpr (sizeof(InT) < sizeof(OutT)) {
      AscendC::Reg::DataCopy<InT, AscendC::Reg::LoadDist::DIST_UNPACK_B32>(srcVreg, srcUb + innerLoopStride * i);
      AscendC::Reg::Cast<OutT, InT, cast_trait_b64>(dstVreg, srcVreg, fullMask);
      AscendC::Reg::DataCopy(dstUb + innerLoopStride * i, dstVreg, b64MaskReg);
    } else {
      AscendC::Reg::DataCopy((AscendC::Reg::RegTensor<int64_t> &)srcVreg,
                             (__ubuf__ int64_t *&)srcUb + innerLoopStride * i);
      AscendC::Reg::Cast<OutT, InT, cast_trait_b64>(tmpoutVreg, srcVreg, fullMask2);
      AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t> &)dstVreg, (AscendC::Reg::RegTensor<InT> &)tmpoutVreg);
      AscendC::Reg::DataCopy((__ubuf__ OutT *&)dstUb + innerLoopStride * i, (AscendC::Reg::RegTensor<OutT> &)dstVreg,
                             b32MaskReg);
    }
  }

  tailMaskReg = AscendC::Reg::UpdateMask<OutT>(tailnum);
  for (uint32_t i = 0; i < tailCtrl; i++) {
    if constexpr (sizeof(InT) < sizeof(OutT)) {
      AscendC::Reg::DataCopy<InT, AscendC::Reg::LoadDist::DIST_UNPACK_B32>(
          srcVreg, srcUb + innerLoopStride * (repeatTimes - 1) + innerLoopStride * i);
      AscendC::Reg::Cast<OutT, InT, cast_trait_b64>(dstVreg, srcVreg, fullMask);
      AscendC::Reg::DataCopy(dstUb + innerLoopStride * (repeatTimes - 1) + innerLoopStride * i, dstVreg, tailMaskReg);
    } else {
      AscendC::Reg::DataCopy((AscendC::Reg::RegTensor<int64_t> &)srcVreg,
                             (__ubuf__ int64_t *&)srcUb + innerLoopStride * i + innerLoopStride * (repeatTimes - 1));
      AscendC::Reg::Cast<OutT, InT, cast_trait_b64>(tmpoutVreg, srcVreg, fullMask2);
      AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t> &)dstVreg, (AscendC::Reg::RegTensor<InT> &)tmpoutVreg);
      AscendC::Reg::DataCopy((__ubuf__ OutT *&)dstUb + innerLoopStride * (repeatTimes - 1) + innerLoopStride * i,
                             (AscendC::Reg::RegTensor<OutT> &)dstVreg, tailMaskReg);
    }
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendB64Transfer(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                              uint32_t repeatTimes, uint32_t innerLoopStride) {
  static constexpr AscendC::Reg::CastTrait cast_trait_in = {
      AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
      GetCastRoundMode<InT, float>()};
  static constexpr AscendC::Reg::CastTrait cast_trait_out = {
      AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
      GetCastRoundMode<float, OutT>()};
  uint32_t sreg = static_cast<uint32_t>(count);
  uint32_t b64Sreg = static_cast<uint32_t>(count * 2);
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  AscendC::Reg::RegTensor<float> midVreg;
  AscendC::Reg::RegTensor<uint32_t> zeroVreg, tmpVreg;
  AscendC::Reg::MaskReg b64MaskReg, b32MaskReg, maskReg;
  constexpr uint8_t elePerBlkInT = GetDataBlockSizeInBytes() / sizeof(InT);
  constexpr uint8_t elePerBlkOutT = GetDataBlockSizeInBytes() / sizeof(OutT);
  AscendC::Reg::MaskReg fullPreg = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
  AscendC::Reg::Duplicate(zeroVreg, 0, fullPreg);

  for (uint32_t i = 0; i < repeatTimes; i++) {
    b32MaskReg = AscendC::Reg::UpdateMask<uint32_t>(sreg);
    b64MaskReg = AscendC::Reg::UpdateMask<uint32_t>(b64Sreg);
    if constexpr (sizeof(InT) < sizeof(float)) {
      AscendC::Reg::MaskPack(maskReg, b32MaskReg);
      GenLoadInstr<InT, float>(srcVreg, srcUb + innerLoopStride * i);
      AscendC::Reg::Cast<float, InT, cast_trait_in>(midVreg, srcVreg, b64MaskReg);

      AscendC::Reg::Interleave((AscendC::Reg::RegTensor<uint32_t> &)midVreg, tmpVreg,
                               (AscendC::Reg::RegTensor<uint32_t> &)midVreg, zeroVreg);
      AscendC::Reg::Cast<OutT, float, cast_trait_out>(dstVreg, midVreg, b64MaskReg);
      AscendC::Reg::DataCopy((__ubuf__ uint32_t *&)dstUb + innerLoopStride * i * 2,
                             (AscendC::Reg::RegTensor<uint32_t> &)dstVreg, b64MaskReg);
    } else {
      AscendC::Reg::DataCopy((AscendC::Reg::RegTensor<uint32_t> &)srcVreg,
                             (__ubuf__ uint32_t *&)srcUb + innerLoopStride * i * 2);
      AscendC::Reg::Cast<float, InT, cast_trait_in>(midVreg, srcVreg, b64MaskReg);
      AscendC::Reg::DeInterleave((AscendC::Reg::RegTensor<uint32_t> &)midVreg, tmpVreg,
                                 (AscendC::Reg::RegTensor<uint32_t> &)midVreg, zeroVreg);

      AscendC::Reg::MaskPack(maskReg, b64MaskReg);
      AscendC::Reg::Cast<OutT, float, cast_trait_out>(dstVreg, midVreg, b64MaskReg);
      GenStoreInstr<float, OutT>(dstUb + innerLoopStride * i, dstVreg, maskReg);
    }
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendInT(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                      uint32_t repeatTimes, uint32_t innerLoopStride, uint32_t mainBlockCount,
                                      uint32_t tailCount, uint32_t offset, uint32_t singleMainBlockCtrl) {
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  AscendC::Reg::MaskReg mainBlockMask, tailBlockMask;
  mainBlockMask = AscendC::Reg::UpdateMask<InT>(mainBlockCount);
  tailBlockMask = AscendC::Reg::UpdateMask<InT>(tailCount);
  for (uint32_t i = 0; i < repeatTimes; i++) {
    GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i);
    GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, (AscendC::Reg::RegTensor<OutT> &)srcVreg, mainBlockMask);
    // unroll
    GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i + offset);
    GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i + offset, (AscendC::Reg::RegTensor<OutT> &)srcVreg,
                             mainBlockMask);
  }
  for (uint16_t j = 0; j < singleMainBlockCtrl; ++j) {
    GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * repeatTimes * 2);
    GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * repeatTimes * 2, (AscendC::Reg::RegTensor<OutT> &)srcVreg,
                             mainBlockMask);
  }
  GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * repeatTimes * 2 + singleMainBlockCtrl * innerLoopStride);
  GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * repeatTimes * 2 + singleMainBlockCtrl * innerLoopStride,
                           (AscendC::Reg::RegTensor<OutT> &)srcVreg, tailBlockMask);
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
__simd_vf__ inline void CastExtendS64U8(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                        uint32_t repeatTimesUnRoll, uint32_t innerLoopStride, uint32_t tailCount,
                                        uint32_t repeatTimes, uint32_t tailBlockCtrl) {
  static constexpr AscendC::Reg::CastTrait cast_trait = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_TRUNC};

  uint32_t sreg = 32;

  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  AscendC::Reg::RegTensor<uint32_t> tmp32Vreg;
  AscendC::Reg::RegTensor<uint16_t> tmp16Vreg;

  AscendC::Reg::MaskReg stMaskReg, tailMaskReg;

  stMaskReg = AscendC::Reg::UpdateMask<OutT>(sreg);

  for (uint32_t i = 0; i < repeatTimes; i++) {
    GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i);
    AscendC::Reg::Pack<uint32_t, int64_t, AscendC::Reg::HighLowPart::LOWEST>(tmp32Vreg, srcVreg);
    AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(tmp16Vreg, tmp32Vreg);
    AscendC::Reg::Pack<uint8_t, uint16_t, AscendC::Reg::HighLowPart::LOWEST>(dstVreg, tmp16Vreg);
    GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
  }

  for (uint32_t i = 0; i < tailBlockCtrl; i++) {
    tailMaskReg = AscendC::Reg::UpdateMask<OutT>(tailCount);
    GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * repeatTimes + innerLoopStride * i);
    AscendC::Reg::Pack<uint32_t, int64_t, AscendC::Reg::HighLowPart::LOWEST>(tmp32Vreg, srcVreg);
    AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(tmp16Vreg, tmp32Vreg);
    AscendC::Reg::Pack<uint8_t, uint16_t, AscendC::Reg::HighLowPart::LOWEST>(dstVreg, tmp16Vreg);
    GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * repeatTimes + innerLoopStride * i, dstVreg, tailMaskReg);
  }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __simd_vf__ void CastExtendU8S64(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                        uint32_t repeatTimes, uint32_t innerLoopStride) {
  uint32_t sreg = static_cast<uint32_t>(count);
  AscendC::Reg::RegTensor<InT> srcVreg;
  AscendC::Reg::RegTensor<uint64_t> tmpU64Vreg;
  AscendC::Reg::RegTensor<OutT> dstVreg;
  AscendC::Reg::MaskReg stMaskReg;

  for (uint32_t i = 0; i < repeatTimes; i++) {
    stMaskReg = AscendC::Reg::UpdateMask<int64_t>(sreg);
    AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(srcVreg, srcUb + innerLoopStride * i);
    AscendC::Reg::UnPack<uint64_t, uint32_t>(tmpU64Vreg, (AscendC::Reg::RegTensor<uint32_t> &)srcVreg);
    AscendC::Reg::DataCopy(dstUb + innerLoopStride * i, (AscendC::Reg::RegTensor<int64_t> &)tmpU64Vreg, stMaskReg);
  }
}

template <auto func, typename InT, typename OutT, AscendC::RoundMode roundMode, uint8_t dim>
__aicore__ inline void CastExtendImpl(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                      uint32_t repeatTimes, uint32_t innerLoopStride,
                                      const uint32_t (&output_dims)[dim], const uint32_t (&output_stride)[dim],
                                      const uint32_t (&input_stride)[dim]) {
  static_assert(dim == 1 || dim == 2, "CastExtend dim exceeds maximum 2");
  if constexpr (dim == 1) {
    func(dstUb, srcUb, count, repeatTimes, innerLoopStride);
  } else if constexpr (dim == 2) {
    uint32_t loop_num = uint32_t(output_dims[0]);
    if ((output_stride[0] == input_stride[0]) && (output_stride[0] == output_dims[1])) {
      uint32_t count_new = output_dims[0] * output_stride[0];
      uint32_t repeat_times_new = static_cast<uint32_t>((count_new + innerLoopStride - 1) / innerLoopStride);
      func(dstUb, srcUb, count_new, repeat_times_new, innerLoopStride);
    } else {
      for (uint32_t i = 0; i < loop_num; i++) {
        if constexpr (std::is_same_v<InT, int4x2_t>) {
          func(dstUb + i * output_stride[0], srcUb + i * input_stride[0] / 2, count, repeatTimes, innerLoopStride);
        } else if constexpr (std::is_same_v<OutT, int4x2_t>) {
          func(dstUb + i * output_stride[0] / 2, srcUb + i * input_stride[0], count, repeatTimes, innerLoopStride);
        } else {
          func(dstUb + i * output_stride[0], srcUb + i * input_stride[0], count, repeatTimes, innerLoopStride);
        }
      }
    }
  }
}

template <auto func, typename InT, typename OutT, AscendC::RoundMode roundMode, uint8_t dim>
__aicore__ inline void CastExtendImplOptimize(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                              uint32_t repeatTimes, uint32_t innerLoopStride,
                                              const uint32_t (&output_dims)[dim], const uint32_t (&output_stride)[dim],
                                              const uint32_t (&input_stride)[dim]) {
  static_assert(dim == 1 || dim == 2, "CastExtend dim exceeds maximum 2");
  const uint32_t mainBlockCount = innerLoopStride;
  uint32_t tailCount = count % innerLoopStride;
  uint32_t repeatTime = count / innerLoopStride;
  if (tailCount == 0 && repeatTime > 0) {
    repeatTime--;
    tailCount += innerLoopStride;
  }
  uint32_t repeatTimesUnRoll = repeatTime / 2;
  uint32_t singleMainBlockCtrl = repeatTime % 2;
  uint32_t offset = repeatTimesUnRoll * innerLoopStride;
  if constexpr (dim == 1) {
    func(dstUb, srcUb, count, repeatTimesUnRoll, innerLoopStride, mainBlockCount, tailCount, offset,
         singleMainBlockCtrl);
  } else if constexpr (dim == 2) {
    uint32_t loop_num = uint32_t(output_dims[0]);
    for (uint32_t i = 0; i < loop_num; i++) {
      if constexpr (std::is_same_v<InT, int4x2_t>) {
        func(dstUb + i * output_stride[0], srcUb + i * input_stride[0] / 2, count, repeatTimesUnRoll, innerLoopStride,
             mainBlockCount, tailCount, offset, singleMainBlockCtrl);
      } else if constexpr (std::is_same_v<OutT, int4x2_t>) {
        func(dstUb + i * output_stride[0] / 2, srcUb + i * input_stride[0], count, repeatTimesUnRoll, innerLoopStride,
             mainBlockCount, tailCount, offset, singleMainBlockCtrl);
      } else {
        func(dstUb + i * output_stride[0], srcUb + i * input_stride[0], count, repeatTimesUnRoll, innerLoopStride,
             mainBlockCount, tailCount, offset, singleMainBlockCtrl);
      }
    }
  }
}

template <auto func, typename InT, typename OutT, AscendC::RoundMode roundMode, uint8_t dim>
__aicore__ inline void CastExtendImplOptimizeS64U8(__ubuf__ OutT *dstUb, __ubuf__ InT *srcUb, const int64_t count,
                                                   uint32_t innerLoopStride, const uint32_t (&output_dims)[dim],
                                                   const uint32_t (&output_stride)[dim],
                                                   const uint32_t (&input_stride)[dim]) {
  static_assert(dim == 1 || dim == 2, "CastExtend dim exceeds maximum 2");
  const uint32_t mainBlockCount = innerLoopStride;
  uint32_t tailCount = count % innerLoopStride;
  uint32_t repeatTimesUnRoll = count / (GetVecLen() / sizeof(InT));
  uint32_t repeatTimes = count / (GetVecLen() / sizeof(InT));
  uint32_t tailBlockCtrl = (tailCount + count - 1) / count;
  if constexpr (dim == 1) {
    func(dstUb, srcUb, count, repeatTimesUnRoll, innerLoopStride, tailCount, repeatTimes, tailBlockCtrl);
  } else if constexpr (dim == 2) {
    uint32_t loop_num = uint32_t(output_dims[0]);
    for (uint32_t i = 0; i < loop_num; i++) {
      func(dstUb + i * output_stride[0], srcUb + i * input_stride[0], count, repeatTimesUnRoll, innerLoopStride,
           tailCount, repeatTimes, tailBlockCtrl);
    }
  }
}

template <typename InT, typename OutT, uint8_t dim, AscendC::RoundMode round_mode = AscendC::RoundMode::UNKNOWN>
__aicore__ inline void CastExtend(const AscendC::LocalTensor<OutT> &dst, const AscendC::LocalTensor<InT> &src,
                                  const uint32_t (&output_dims)[dim], const uint32_t (&output_stride)[dim],
                                  const uint32_t (&input_stride)[dim]) {
  constexpr AscendC::RoundMode roundMode =
      (round_mode == AscendC::RoundMode::UNKNOWN) ? GetCastRoundMode<InT, OutT>() : round_mode;
  __ubuf__ InT *srcUb = (__ubuf__ InT *)src.GetPhyAddr();
  __ubuf__ OutT *dstUb = (__ubuf__ OutT *)dst.GetPhyAddr();

  uint32_t count = output_dims[dim - 1];
  constexpr uint32_t repeatStrideSrc = static_cast<uint32_t>(GetVecLen() / sizeof(InT));
  constexpr uint32_t repeatStrideDst = static_cast<uint32_t>(GetVecLen() / sizeof(OutT));
  uint32_t innerLoopStride = repeatStrideSrc > repeatStrideDst ? repeatStrideDst : repeatStrideSrc;
  uint32_t repeatTimes = static_cast<uint32_t>((count + innerLoopStride - 1) / innerLoopStride);

  constexpr bool b4Cast = SupportType<Tuple<OutT, InT>, Tuple<half, int4x2_t>, Tuple<int4x2_t, half>>();

  constexpr bool floatBoolCast = SupportType<Tuple<OutT, InT>, Tuple<bool, float>>();
  constexpr bool int64BoolCast = SupportType<Tuple<OutT, InT>, Tuple<bool, int64_t>>();
  constexpr bool int32BoolCast = SupportType<Tuple<OutT, InT>, Tuple<bool, int32_t>>();
  constexpr bool smallIntBoolCast = SupportType<Tuple<OutT, InT>, Tuple<bool, int16_t>, Tuple<bool, int8_t>>();

  constexpr bool b8Cast =
      AscendC::IsSameType<InT, uint8_t>::value && AscendC::SupportType<OutT, float, int32_t, int16_t, int4x2_t>();

  constexpr bool b64Cast = SupportType<Tuple<OutT, InT>, Tuple<float, int64_t>, Tuple<int64_t, float>,
                                       Tuple<int32_t, int64_t>, Tuple<int64_t, int32_t>>();

  constexpr bool b64CastWithTransfer =
      SupportType<Tuple<OutT, InT>, Tuple<half, int64_t>, Tuple<int64_t, half>, Tuple<bfloat16_t, int64_t>>();

  constexpr bool castWithSameBit =
      SupportType<Tuple<OutT, InT>, Tuple<uint8_t, int8_t>, Tuple<int8_t, uint8_t>, Tuple<uint16_t, int16_t>,
                  Tuple<int16_t, uint16_t>, Tuple<uint32_t, int32_t>, Tuple<int32_t, uint32_t>,
                  Tuple<uint64_t, int64_t>, Tuple<int64_t, uint64_t>>();

  constexpr bool s64U8Cast = SupportType<Tuple<OutT, InT>, Tuple<uint8_t, int64_t>>();

  int64_t ctrl60_value = AscendC::GetCtrlSpr<60, 60>();
  int64_t ctrl48_value = AscendC::GetCtrlSpr<48, 48>();
  int64_t ctrl59_value = AscendC::GetCtrlSpr<59, 59>();

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
  AscendC::SetCtrlSpr<60, 60>(1);
  AscendC::SetCtrlSpr<48, 48>(0);
  AscendC::SetCtrlSpr<59, 59>(0);
#endif

  if constexpr (floatBoolCast) {
    constexpr auto func = CastExtendFloatBool<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (int64BoolCast) {
    constexpr auto func = CastExtendInt64Bool<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (int32BoolCast) {
    constexpr auto func = CastExtendInt32Bool<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (smallIntBoolCast) {
    constexpr auto func = CastExtendSmallIntBool<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (b4Cast) {
    constexpr auto func = CastExtendB4<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (b8Cast) {
    if constexpr (AscendC::IsSameType<InT, uint8_t>::value && (AscendC::SupportType<OutT, int4x2_t>())) {
      innerLoopStride = static_cast<uint32_t>(GetVecLen() / sizeof(half));
      repeatTimes = static_cast<uint32_t>((count + innerLoopStride - 1) / innerLoopStride);
    }
    constexpr auto func = CastExtendB8<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (b64CastWithTransfer) {
    constexpr auto func = CastExtendB64Transfer<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (b64Cast) {
    constexpr auto func = CastExtendB64<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (castWithSameBit) {
    constexpr auto func = CastExtendInT<InT, OutT, roundMode>;
    CastExtendImplOptimize<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride,
                                                            output_dims, output_stride, input_stride);
  } else if constexpr (s64U8Cast) {
    constexpr auto func = CastExtendS64U8<InT, OutT, roundMode>;
    CastExtendImplOptimizeS64U8<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, innerLoopStride, output_dims,
                                                                 output_stride, input_stride);
  } else if constexpr (SupportType<Tuple<OutT, InT>, Tuple<float, int8_t>>()) {
    constexpr auto func = CastExtendInt8ToFloat<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (SupportType<Tuple<OutT, InT>, Tuple<int8_t, float>, Tuple<uint8_t, float>>()) {
    constexpr auto func = CastExtendFloatToInt8<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (SupportType<Tuple<OutT, InT>, Tuple<int16_t, int8_t>>()) {
    constexpr auto func = CastExtendInt8ToInt16<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (SupportType<Tuple<OutT, InT>, Tuple<int8_t, int16_t>>()) {
    constexpr auto func = CastExtendInt16ToInt8<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (SupportType<Tuple<OutT, InT>, Tuple<uint8_t, int16_t>>()) {
#if defined(__NPU_ARCH__) && \
    ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    AscendC::SetCtrlSpr<60, 60>(0);
#endif
    constexpr auto func = CastExtendInt16ToUint8<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else if constexpr (SupportType<Tuple<OutT, InT>, Tuple<int64_t, uint8_t>>()) {
    constexpr auto func = CastExtendU8S64<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  } else {
#if defined(__NPU_ARCH__) && \
    ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    AscendC::SetCtrlSpr<60, 60>(0);
#endif
    constexpr auto func = CastExtendCommon<InT, OutT, roundMode>;
    CastExtendImpl<func, InT, OutT, roundMode, dim>(dstUb, srcUb, count, repeatTimes, innerLoopStride, output_dims,
                                                    output_stride, input_stride);
  }

  AscendC::SetCtrlSpr<60, 60>(ctrl60_value);
  AscendC::SetCtrlSpr<48, 48>(ctrl48_value);
  AscendC::SetCtrlSpr<59, 59>(ctrl59_value);
}

template <typename InT, typename OutT, uint8_t dim>
__aicore__ inline void Ceil2IntExtend(const AscendC::LocalTensor<OutT> &dst, const AscendC::LocalTensor<InT> &src,
                                      const uint32_t (&output_dims)[dim], const uint32_t (&output_stride)[dim],
                                      const uint32_t (&input_stride)[dim]) {
  CastExtend<InT, OutT, dim, AscendC::RoundMode::CAST_CEIL>(dst, src, output_dims, output_stride, input_stride);
}
#endif
