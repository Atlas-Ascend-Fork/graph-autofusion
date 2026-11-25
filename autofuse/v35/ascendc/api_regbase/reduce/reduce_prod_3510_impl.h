/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_REDUCE_EXTEND_REDUCE_PROD_REDUCE_PROD_C310_IMPL_H_
#define AUTOFUSE_REDUCE_EXTEND_REDUCE_PROD_REDUCE_PROD_C310_IMPL_H_

#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
#include "reduce/reduce_common_util_3510_impl.h"
#include "reduce/reduce_common_ar_reuse_align_3510_impl.h"
#include "reduce/reduce_common_ra_reuse_align_3510_impl.h"
#include "reduce/reduce_common_ar_ra_reuse_unalign_3510_impl.h"
#ifdef ASCENDC_CPU_DEBUG
#endif  // ASCENDC_CPU_DEBUG

namespace AscendC {
namespace ReduceExtendInternal {
namespace Internal {
template <class T>
__simd_vf__ inline void ReduceProdCopyUnaligned(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t calCount) {
  constexpr uint16_t vecElemNum = GetVecLen() / sizeof(T);
  Reg::RegTensor<T> vreg;
  Reg::MaskReg mask;
  for (uint32_t offset = 0; offset < calCount; offset += vecElemNum) {
    uint32_t count = calCount - offset < vecElemNum ? calCount - offset : vecElemNum;
    mask = Reg::UpdateMask<T>(count);
    Reg::LoadAlign(vreg, srcAddr + offset);
    Reg::StoreAlign(dstAddr + offset, vreg, mask);
  }
  Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <class T, const Reg::RegTrait &Trait, auto Binaryfunc>
__simd_vf__ inline void ReduceProdRAOverVLCopySource(__ubuf__ T *tmpAddr, __ubuf__ T *srcAddr, uint16_t dimA,
                                                     uint32_t mainR, uint32_t tailR, uint16_t loopANum) {
  constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
  Reg::RegTensor<T, Trait> vregTmp;
  Reg::MaskReg mask;
  for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
    uint32_t maskScalar = dimA - loopA * vlSize;
    mask = Reg::UpdateMask<T, Trait>(maskScalar);
    for (uint16_t loopR = static_cast<uint16_t>(tailR); loopR < static_cast<uint16_t>(mainR); loopR++) {
      Reg::LoadAlign(vregTmp, srcAddr + loopA * vlSize + loopR * dimA);
      Reg::StoreAlign(tmpAddr + loopA * vlSize + loopR * dimA, vregTmp, mask);
    }
  }
}

template <class T, const Reg::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__simd_vf__ inline void ReduceProdRAOverVLMergeTail(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *addr,
                                                    uint16_t dimA, uint32_t mainR, uint32_t tailR, uint16_t loopANum,
                                                    uint16_t noNeedCopy) {
  constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
  const uint32_t aTailOffset = mainR * dimA;
  Reg::RegTensor<T, Trait> vregMain;
  Reg::RegTensor<T, Trait> vregTail;
  Reg::MaskReg mask;
  for (uint16_t i = 0; i < noNeedCopy; ++i) {
    for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
      uint32_t maskScalar = dimA - loopA * vlSize;
      mask = Reg::UpdateMask<T, Trait>(maskScalar);
      Reg::LoadAlign(vregMain, srcAddr + loopA * vlSize);
      Reg::LoadAlign(vregTail, srcAddr + loopA * vlSize + aTailOffset);
      Binaryfunc(vregMain, vregMain, vregTail, mask);
      Reg::StoreAlign(dstAddr + loopA * vlSize, vregMain, mask);
    }
    return;
  }
  for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
    uint32_t maskScalar = dimA - loopA * vlSize;
    mask = Reg::UpdateMask<T, Trait>(maskScalar);
    for (uint16_t loopR = 0; loopR < static_cast<uint16_t>(tailR); loopR++) {
      Reg::LoadAlign(vregMain, srcAddr + loopA * vlSize + loopR * dimA);
      Reg::LoadAlign(vregTail, srcAddr + loopA * vlSize + aTailOffset + loopR * dimA);
      Binaryfunc(vregMain, vregMain, vregTail, mask);
      Reg::StoreAlign(addr + loopA * vlSize + loopR * dimA, vregMain, mask);
    }
  }
  Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <class T, const Reg::RegTrait &Trait>
__simd_callee__ inline void ReduceProdRAOverVLLoadVregs(
    Reg::RegTensor<T, Trait> &vreg0, Reg::RegTensor<T, Trait> &vreg1, Reg::RegTensor<T, Trait> &vreg2,
    Reg::RegTensor<T, Trait> &vreg3, Reg::RegTensor<T, Trait> &vreg4, Reg::RegTensor<T, Trait> &vreg5,
    Reg::RegTensor<T, Trait> &vreg6, Reg::RegTensor<T, Trait> &vreg7, Reg::RegTensor<T, Trait> &vreg8,
    Reg::RegTensor<T, Trait> &vreg9, Reg::RegTensor<T, Trait> &vreg10, Reg::RegTensor<T, Trait> &vreg11,
    Reg::RegTensor<T, Trait> &vreg12, Reg::RegTensor<T, Trait> &vreg13, Reg::RegTensor<T, Trait> &vreg14,
    Reg::RegTensor<T, Trait> &vreg15, __ubuf__ T *addr, uint32_t offsetR, uint32_t baseOffset) {
  Reg::LoadAlign(vreg0, addr + baseOffset);
  Reg::LoadAlign(vreg1, addr + offsetR + baseOffset);
  Reg::LoadAlign(vreg2, addr + 2 * offsetR + baseOffset);
  Reg::LoadAlign(vreg3, addr + 3 * offsetR + baseOffset);
  Reg::LoadAlign(vreg4, addr + 4 * offsetR + baseOffset);
  Reg::LoadAlign(vreg5, addr + 5 * offsetR + baseOffset);
  Reg::LoadAlign(vreg6, addr + 6 * offsetR + baseOffset);
  Reg::LoadAlign(vreg7, addr + 7 * offsetR + baseOffset);
  Reg::LoadAlign(vreg8, addr + 8 * offsetR + baseOffset);
  Reg::LoadAlign(vreg9, addr + 9 * offsetR + baseOffset);
  Reg::LoadAlign(vreg10, addr + 10 * offsetR + baseOffset);
  Reg::LoadAlign(vreg11, addr + 11 * offsetR + baseOffset);
  Reg::LoadAlign(vreg12, addr + 12 * offsetR + baseOffset);
  Reg::LoadAlign(vreg13, addr + 13 * offsetR + baseOffset);
  Reg::LoadAlign(vreg14, addr + 14 * offsetR + baseOffset);
  Reg::LoadAlign(vreg15, addr + 15 * offsetR + baseOffset);
}

template <class T, const Reg::RegTrait &Trait, auto Binaryfunc>
__simd_callee__ inline void ReduceProdRAOverVLFoldVregs(
    Reg::RegTensor<T, Trait> &vreg0, Reg::RegTensor<T, Trait> &vreg1, Reg::RegTensor<T, Trait> &vreg2,
    Reg::RegTensor<T, Trait> &vreg3, Reg::RegTensor<T, Trait> &vreg4, Reg::RegTensor<T, Trait> &vreg5,
    Reg::RegTensor<T, Trait> &vreg6, Reg::RegTensor<T, Trait> &vreg7, Reg::RegTensor<T, Trait> &vreg8,
    Reg::RegTensor<T, Trait> &vreg9, Reg::RegTensor<T, Trait> &vreg10, Reg::RegTensor<T, Trait> &vreg11,
    Reg::RegTensor<T, Trait> &vreg12, Reg::RegTensor<T, Trait> &vreg13, Reg::RegTensor<T, Trait> &vreg14,
    Reg::RegTensor<T, Trait> &vreg15, Reg::MaskReg mask) {
  Binaryfunc(vreg0, vreg0, vreg8, mask);
  Binaryfunc(vreg1, vreg1, vreg9, mask);
  Binaryfunc(vreg2, vreg2, vreg10, mask);
  Binaryfunc(vreg3, vreg3, vreg11, mask);
  Binaryfunc(vreg4, vreg4, vreg12, mask);
  Binaryfunc(vreg5, vreg5, vreg13, mask);
  Binaryfunc(vreg6, vreg6, vreg14, mask);
  Binaryfunc(vreg7, vreg7, vreg15, mask);
  Binaryfunc(vreg0, vreg0, vreg4, mask);
  Binaryfunc(vreg1, vreg1, vreg5, mask);
  Binaryfunc(vreg2, vreg2, vreg6, mask);
  Binaryfunc(vreg3, vreg3, vreg7, mask);
  Binaryfunc(vreg0, vreg0, vreg2, mask);
  Binaryfunc(vreg1, vreg1, vreg3, mask);
  Binaryfunc(vreg0, vreg0, vreg1, mask);
}

template <class T, const Reg::RegTrait &Trait, auto Binaryfunc>
__simd_vf__ inline void ReduceProdRAOverVLFoldMain(__ubuf__ T *addr, uint16_t dimA, uint32_t mainR, uint16_t loopANum,
                                                   uint16_t mainTimes, uint16_t avgFolds) {
  constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
  Reg::RegTensor<T, Trait> vreg0;
  Reg::RegTensor<T, Trait> vreg1;
  Reg::RegTensor<T, Trait> vreg2;
  Reg::RegTensor<T, Trait> vreg3;
  Reg::RegTensor<T, Trait> vreg4;
  Reg::RegTensor<T, Trait> vreg5;
  Reg::RegTensor<T, Trait> vreg6;
  Reg::RegTensor<T, Trait> vreg7;
  Reg::RegTensor<T, Trait> vreg8;
  Reg::RegTensor<T, Trait> vreg9;
  Reg::RegTensor<T, Trait> vreg10;
  Reg::RegTensor<T, Trait> vreg11;
  Reg::RegTensor<T, Trait> vreg12;
  Reg::RegTensor<T, Trait> vreg13;
  Reg::RegTensor<T, Trait> vreg14;
  Reg::RegTensor<T, Trait> vreg15;
  Reg::MaskReg mask;
  uint32_t maskScalar = 0;
  uint16_t loopRNum = mainR;
  for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
    loopRNum = loopRNum >> avgFolds;
    uint16_t offsetR = loopRNum * dimA;
    for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
      maskScalar = dimA - loopA * vlSize;
      mask = Reg::UpdateMask<T, Trait>(maskScalar);
      for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
        const uint32_t baseOffset = loopA * vlSize + loopR * dimA;
        ReduceProdRAOverVLLoadVregs<T, Trait>(vreg0, vreg1, vreg2, vreg3, vreg4, vreg5, vreg6, vreg7, vreg8, vreg9,
                                              vreg10, vreg11, vreg12, vreg13, vreg14, vreg15, addr, offsetR,
                                              baseOffset);
        ReduceProdRAOverVLFoldVregs<T, Trait, Binaryfunc>(vreg0, vreg1, vreg2, vreg3, vreg4, vreg5, vreg6, vreg7, vreg8,
                                                          vreg9, vreg10, vreg11, vreg12, vreg13, vreg14, vreg15, mask);
        Reg::StoreAlign(addr + baseOffset, vreg0, mask);
      }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
  }
}

template <class T, const Reg::RegTrait &Trait, auto Binaryfunc>
__simd_vf__ inline void ReduceProdRAOverVLFoldOne(__ubuf__ T *dstAddr, __ubuf__ T *addr, uint16_t dimA,
                                                  uint16_t loopANum) {
  constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
  Reg::RegTensor<T, Trait> vreg0;
  Reg::RegTensor<T, Trait> vreg1;
  Reg::MaskReg mask;
  for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
    uint32_t maskScalar = dimA - loopA * vlSize;
    mask = Reg::UpdateMask<T, Trait>(maskScalar);
    Reg::LoadAlign(vreg0, addr + loopA * vlSize);
    Reg::LoadAlign(vreg1, addr + dimA + loopA * vlSize);
    Binaryfunc(vreg0, vreg0, vreg1, mask);
    Reg::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
  }
}

template <class T, const Reg::RegTrait &Trait, auto Binaryfunc>
__simd_vf__ inline void ReduceProdRAOverVLFoldTwo(__ubuf__ T *dstAddr, __ubuf__ T *addr, uint16_t dimA,
                                                  uint16_t loopANum) {
  constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
  Reg::RegTensor<T, Trait> vreg0;
  Reg::RegTensor<T, Trait> vreg1;
  Reg::RegTensor<T, Trait> vreg2;
  Reg::RegTensor<T, Trait> vreg3;
  Reg::MaskReg mask;
  for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
    uint32_t maskScalar = dimA - loopA * vlSize;
    mask = Reg::UpdateMask<T, Trait>(maskScalar);
    Reg::LoadAlign(vreg0, addr + loopA * vlSize);
    Reg::LoadAlign(vreg1, addr + dimA + loopA * vlSize);
    Reg::LoadAlign(vreg2, addr + 2 * dimA + loopA * vlSize);
    Reg::LoadAlign(vreg3, addr + 3 * dimA + loopA * vlSize);
    Binaryfunc(vreg0, vreg0, vreg2, mask);
    Binaryfunc(vreg1, vreg1, vreg3, mask);
    Binaryfunc(vreg0, vreg0, vreg1, mask);
    Reg::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
  }
}

template <class T, const Reg::RegTrait &Trait, auto Binaryfunc>
__simd_vf__ inline void ReduceProdRAOverVLFoldThree(__ubuf__ T *dstAddr, __ubuf__ T *addr, uint16_t dimA,
                                                    uint16_t loopANum) {
  constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
  Reg::RegTensor<T, Trait> vreg0;
  Reg::RegTensor<T, Trait> vreg1;
  Reg::RegTensor<T, Trait> vreg2;
  Reg::RegTensor<T, Trait> vreg3;
  Reg::RegTensor<T, Trait> vreg4;
  Reg::RegTensor<T, Trait> vreg5;
  Reg::RegTensor<T, Trait> vreg6;
  Reg::RegTensor<T, Trait> vreg7;
  Reg::MaskReg mask;
  for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
    uint32_t maskScalar = dimA - loopA * vlSize;
    mask = Reg::UpdateMask<T, Trait>(maskScalar);
    Reg::LoadAlign(vreg0, addr + loopA * vlSize);
    Reg::LoadAlign(vreg1, addr + dimA + loopA * vlSize);
    Reg::LoadAlign(vreg2, addr + 2 * dimA + loopA * vlSize);
    Reg::LoadAlign(vreg3, addr + 3 * dimA + loopA * vlSize);
    Reg::LoadAlign(vreg4, addr + 4 * dimA + loopA * vlSize);
    Reg::LoadAlign(vreg5, addr + 5 * dimA + loopA * vlSize);
    Reg::LoadAlign(vreg6, addr + 6 * dimA + loopA * vlSize);
    Reg::LoadAlign(vreg7, addr + 7 * dimA + loopA * vlSize);
    Binaryfunc(vreg0, vreg0, vreg4, mask);
    Binaryfunc(vreg1, vreg1, vreg5, mask);
    Binaryfunc(vreg2, vreg2, vreg6, mask);
    Binaryfunc(vreg3, vreg3, vreg7, mask);
    Binaryfunc(vreg0, vreg0, vreg2, mask);
    Binaryfunc(vreg1, vreg1, vreg3, mask);
    Binaryfunc(vreg0, vreg0, vreg1, mask);
    Reg::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
  }
}

template <class T, const Reg::RegTrait &Trait, auto Binaryfunc>
__simd_vf__ inline void ReduceProdRAOverVLFoldZero(__ubuf__ T *dstAddr, __ubuf__ T *addr, uint16_t dimA,
                                                   uint16_t loopANum) {
  constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
  Reg::RegTensor<T, Trait> vreg0;
  Reg::MaskReg mask;
  for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
    uint32_t maskScalar = dimA - loopA * vlSize;
    mask = Reg::UpdateMask<T, Trait>(maskScalar);
    Reg::LoadAlign(vreg0, addr + loopA * vlSize);
    Reg::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
  }
}

template <class T, const Reg::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__aicore__ inline void ReduceProdRAOverVLVF(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
                                            uint16_t dimA, uint32_t dimR, uint32_t mainR, uint32_t tailR,
                                            uint16_t loopANum, uint16_t folds, uint16_t avgFolds, uint16_t foldZero,
                                            uint16_t foldOne, uint16_t foldTwo, uint16_t foldThree) {
  uint16_t noNeedCopy = mainR > 1 ? 0 : 1;
  uint16_t mainTimes = folds / avgFolds;

  __ubuf__ T *addr;
  if constexpr (!isReuseSource) {
    ReduceProdRAOverVLCopySource<T, Trait, Binaryfunc>(tmpAddr, srcAddr, dimA, mainR, tailR, loopANum);
    addr = tmpAddr;
  } else {
    addr = srcAddr;
  }
  ReduceProdRAOverVLMergeTail<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, addr, dimA, mainR, tailR, loopANum,
                                                                   noNeedCopy);
  if (noNeedCopy == 1U) {
    return;
  }
  ReduceProdRAOverVLFoldMain<T, Trait, Binaryfunc>(addr, dimA, mainR, loopANum, mainTimes, avgFolds);
  if (foldOne == 1U) {
    ReduceProdRAOverVLFoldOne<T, Trait, Binaryfunc>(dstAddr, addr, dimA, loopANum);
  } else if (foldTwo == 1U) {
    ReduceProdRAOverVLFoldTwo<T, Trait, Binaryfunc>(dstAddr, addr, dimA, loopANum);
  } else if (foldThree == 1U) {
    ReduceProdRAOverVLFoldThree<T, Trait, Binaryfunc>(dstAddr, addr, dimA, loopANum);
  } else if (foldZero == 1U) {
    ReduceProdRAOverVLFoldZero<T, Trait, Binaryfunc>(dstAddr, addr, dimA, loopANum);
  }
}

template <class T, const Reg::RegTrait &Trait, auto Binaryfunc, bool isReuseSource>
__aicore__ inline void ReduceProdRAOverVL(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr, uint32_t dimA,
                                          uint32_t dimR) {
  constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
  uint32_t mainR = ReduceOpInternal::CalculateMainR(dimR, false, vlSize);
  uint32_t tailR = dimR - mainR;
  uint16_t loopANum = CeilDivision(static_cast<uint16_t>(dimA), vlSize);

  if (mainR == 1) {
    ReduceOpInternal::ReduceCopyOutImpl<T>(dstAddr, srcAddr, dimA);
    return;
  }
  if constexpr (!isReuseSource) {
    if (tailR == 0 && mainR > 1) {
      mainR = mainR / 2;
      tailR = mainR;
    }
  }
  uint16_t folds = ReduceOpInternal::CalculateFolds(mainR);
  uint16_t avgFolds = ReduceOpInternal::BASE_FOLD;
  uint16_t tailFolds = folds % avgFolds;
  uint16_t foldZero = (tailFolds == 0) ? 1 : 0;
  uint16_t foldOne = (tailFolds == ReduceOpInternal::FOLD_ONE) ? 1 : 0;
  uint16_t foldTwo = (tailFolds == ReduceOpInternal::FOLD_TWO) ? 1 : 0;
  uint16_t foldThree = (tailFolds == ReduceOpInternal::FOLD_THREE) ? 1 : 0;

  ReduceProdRAOverVLVF<T, Trait, Binaryfunc, isReuseSource>(dstAddr, srcAddr, tmpAddr, static_cast<uint16_t>(dimA),
                                                            dimR, mainR, tailR, loopANum, folds, avgFolds, foldZero,
                                                            foldOne, foldTwo, foldThree);
}

template <typename T, const Reg::RegTrait &Trait>
__aicore__ inline void ReduceProd(Reg::RegTensor<T, Trait> &dst, Reg::RegTensor<T, Trait> src, Reg::MaskReg mask) {
  Reg::RegTensor<T, Trait> tempOne;
  // mask invalid data in src to one
  Reg::MaskReg fullMask = Reg::CreateMask<T, Reg::MaskPattern::ALL, Trait>();
  Reg::Duplicate(tempOne, 1);
  Reg::Select(src, src, tempOne, mask);

  constexpr uint16_t vecElemNum = GetVecLen() / sizeof(T) * Trait.REG_NUM;
  for (uint16_t elemNum = vecElemNum / 2U; elemNum > 1U; elemNum /= 2U) {
    Reg::DeInterleave(dst, src, src, tempOne);
    Reg::Mul(src, dst, src, fullMask);
  }
  Reg::DeInterleave(dst, src, src, tempOne);
  Reg::Mul(dst, dst, src, fullMask);
}

template <class T, const Reg::RegTrait &Trait, const uint16_t vlSize>
__simd_vf__ inline void ReduceProdARLessThanVLVF(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, uint32_t dimA, uint32_t dimR,
                                                 const uint16_t repeatTime) {
  Reg::RegTensor<T, Trait> vreg0;
  Reg::RegTensor<T, Trait> vreg1;
  Reg::UnalignReg uDst;
  Reg::RegTensor<T, Trait> tempOne;
  // mask invalid data in src to one
  Reg::Duplicate(tempOne, 1);
  uint32_t sreg1 = dimR;
  Reg::MaskReg mask = Reg::UpdateMask<T, Trait>(sreg1);
  for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
    Reg::LoadAlign(vreg0, srcAddr + loopA * dimR);
    Reg::Select(vreg0, vreg0, tempOne, mask);
    if constexpr (sizeof(T) == 1) {
      // fold to 128
      Reg::DeInterleave(vreg1, vreg0, vreg0, tempOne);
      Reg::Mul(vreg0, vreg1, vreg0, mask);
    }
    if constexpr (sizeof(T) <= 2) {
      // fold to 64
      Reg::DeInterleave(vreg1, vreg0, vreg0, tempOne);
      Reg::Mul(vreg0, vreg1, vreg0, mask);
    }
    for (uint16_t i = 0; i < repeatTime; ++i) {
      Reg::DeInterleave(vreg1, vreg0, vreg0, tempOne);
      Reg::Mul(vreg0, vreg1, vreg0, mask);
    }
    // fold to 1
    Reg::DeInterleave(vreg1, vreg0, vreg0, tempOne);
    Reg::Mul(vreg1, vreg1, vreg0, mask);
    Reg::StoreUnAlign((__ubuf__ T *&)dstAddr, vreg1, uDst, 1);
  }
  Reg::StoreUnAlignPost((__ubuf__ T *&)dstAddr, uDst, 0);
}

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceProdARImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr, uint32_t dimA,
                                        uint32_t dimR) {
  constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
  if (dimR == 1) {
    ReduceOpInternal::ReduceCopyOutImpl<T>(dstAddr, srcAddr, dimA);
  } else if (dimR <= vlSize) {
    uint16_t repeatTime = AscendC::Internal::FindClosestPowerOfTwo(dimR);
    repeatTime = (1 << repeatTime) < dimR ? repeatTime : repeatTime - 1;
    ReduceProdARLessThanVLVF<T, Reg::RegTraitNumOne, vlSize>(dstAddr, srcAddr, dimA, dimR, repeatTime);
  } else if (dimR % vlSize != 0) {
    if constexpr (!isReuseSource) {
      ReduceProdCopyUnaligned<T>(tmpAddr, srcAddr, dimA * dimR);
      srcAddr = tmpAddr;
    }
    ReduceARReuseSourceUnAligned<T, Reg::RegTraitNumOne, Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
                                 ReduceProd<T, Reg::RegTraitNumOne>>(dstAddr, srcAddr, dimA, dimR);
  } else {
    ReduceAROverVLImpl<T, Reg::RegTraitNumOne, vlSize, Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
                       ReduceProd<T, Reg::RegTraitNumOne>, isReuseSource>(dstAddr, srcAddr, tmpAddr, dimA, dimR);
  }
}

template <class T, const Reg::RegTrait &Trait, auto Binaryfunc, auto Reducefunc>
__simd_vf__ inline void ReduceProdAROverVLReuseSourceUnAlignedB64VF(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr,
                                                                    uint32_t dimA, uint32_t dimR) {
  constexpr uint16_t vecElemNum = GetVecLen() / sizeof(T);
  Reg::UnalignReg uDst;

  Reg::RegTensor<T, Trait> vreg0;
  Reg::RegTensor<T, Trait> vreg1;
  Reg::RegTensor<T, Trait> accumulator;
  Reg::RegTensor<T, Trait> tempOne;
  Reg::UnalignReg unalignedSrc;

  Reg::Duplicate(accumulator, 1);
  Reg::Duplicate(tempOne, 1);
  for (uint16_t loopA = 0; loopA < static_cast<uint16_t>(dimA); loopA++) {
    Reg::Duplicate(accumulator, 1);
    __ubuf__ T *src = srcAddr + loopA * dimR;
    for (uint32_t offset = 0; offset < dimR; offset += vecElemNum) {
      uint32_t count = dimR - offset < vecElemNum ? dimR - offset : vecElemNum;
      Reg::MaskReg mask = Reg::UpdateMask<T, Trait>(count);
      __ubuf__ T *loadAddr = src + offset;
      Reg::LoadUnAlignPre(unalignedSrc, loadAddr);
      Reg::LoadUnAlign(vreg0, unalignedSrc, loadAddr, count);
      Reg::Select(vreg0, vreg0, tempOne, mask);
      Reg::MaskReg allMask = Reg::CreateMask<T, Reg::MaskPattern::ALL, Trait>();
      Reducefunc(vreg1, vreg0, allMask);
      Binaryfunc(accumulator, accumulator, vreg1, allMask);
    }
    StoreUnAlign((__ubuf__ T *&)dstAddr, accumulator, uDst, 1);
  }
  Reg::StoreUnAlignPost((__ubuf__ T *&)dstAddr, uDst, 0);
}
template <class T, const Reg::RegTrait &Trait, auto Binaryfunc, auto Reducefunc>
__aicore__ inline void ReduceProdAROverVLReuseSourceUnAlignedB64(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr,
                                                                 uint32_t dimA, uint32_t dimR) {
  ReduceProdAROverVLReuseSourceUnAlignedB64VF<T, Trait, Binaryfunc, Reducefunc>(dstAddr, srcAddr, dimA, dimR);
}

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceProdARB64ReuseSourceCompute(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
                                                         const uint32_t srcShape[]) {
  constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
  if ((srcShape[1] * sizeof(T)) % 32 == 0 && srcShape[1] % vlSize == 0) {
    ReduceARImpl<T, Reg::RegTraitNumTwo,
                 Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
                 ReduceProd<T, Reg::RegTraitNumTwo>, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[0],
                                                                    srcShape[1]);
  } else if (srcShape[1] < vlSize) {
    if constexpr (!isReuseSource) {
      ReduceProdCopyUnaligned<T>(tmpAddr, srcAddr, srcShape[0] * srcShape[1]);
      srcAddr = tmpAddr;
    }
    ReduceARReuseSourceUnAligned<T, Reg::RegTraitNumTwo,
                                 Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
                                 ReduceProd<T, Reg::RegTraitNumTwo>>(dstAddr, srcAddr, srcShape[0], srcShape[1]);
  } else {
    ReduceProdAROverVLReuseSourceUnAlignedB64<
        T, Reg::RegTraitNumTwo, Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
        ReduceProd<T, Reg::RegTraitNumTwo>>(dstAddr, srcAddr, srcShape[0], srcShape[1]);
  }
}

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceProdRAB64ReuseSourceCompute(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
                                                         const uint32_t srcShape[]) {
  if ((srcShape[1] * sizeof(T)) % 32 == 0) {
    ReduceRAB64ReuseSource<T, Reg::RegTraitNumTwo,
                           Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
                           isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[1], srcShape[0]);
  } else {
    if constexpr (!isReuseSource) {
      ReduceProdCopyUnaligned<T>(tmpAddr, srcAddr, srcShape[1] * srcShape[0]);
      srcAddr = tmpAddr;
    }
    ReduceRAReuseSourceUnAlignedB64<T, Reg::RegTraitNumTwo,
                                    Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>>(
        dstAddr, srcAddr, srcShape[1], srcShape[0]);
  }
}

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceProdRAReuseSourceCompute(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
                                                      const uint32_t srcShape[]) {
  if (srcShape[0] == 1 || (srcShape[1] * sizeof(T)) % 32 != 0) {
    if ((srcShape[1] * sizeof(T)) % 32 == 0 || srcShape[1] == 1) {
      ReduceProdRAOverVL<T, Reg::RegTraitNumOne, Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
                         isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[1], srcShape[0]);
    } else {
      if constexpr (!isReuseSource) {
        ReduceProdCopyUnaligned<T>(tmpAddr, srcAddr, srcShape[1] * srcShape[0]);
        srcAddr = tmpAddr;
      }
      ReduceRAReuseSourceUnAligned<T, Reg::RegTraitNumOne, Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>>(
          dstAddr, srcAddr, srcShape[1], srcShape[0]);
    }
  } else {
    if (srcShape[1] * sizeof(T) <= GetVecLen() / 2) {
      ReduceRAConcatImpl<T, Reg::RegTraitNumTwo,
                         Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
                         isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[1], srcShape[0]);
    } else if (srcShape[1] > GetVecLen() / sizeof(T) && srcShape[1] <= ReduceOpInternal::U16_STRIDE) {
      ReduceProdRAOverVL<T, Reg::RegTraitNumOne, Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
                         isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[1], srcShape[0]);
    } else {
      ReduceRAImpl<T, Reg::RegTraitNumOne, Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>, isReuseSource>(
          dstAddr, srcAddr, tmpAddr, srcShape[1], srcShape[0]);
    }
  }
}

template <typename T, typename pattern, bool isReuseSource = false>
__aicore__ inline void ReduceProdImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                      const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t srcShape[],
                                      bool srcInnerPad) {
  CHECK_FUNC_HIGHLEVEL_API(ReduceProd, (T, pattern), (dst, src, sharedTmpBuffer, srcShape, srcInnerPad, srcShape[1]));

  CheckTensorPos<T>(dst, Hardware::UB, "dst", "VECIN/VECCALC/VECOUT", "ReduceProd");
  CheckTensorPos<T>(src, Hardware::UB, "src", "VECIN/VECCALC/VECOUT", "ReduceProd");
  CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN/VECCALC/VECOUT", "ReduceProd");
  static_assert(SupportType<T, int16_t, int32_t, int64_t, float>(),
                "ReduceProd only support int16_t/int32_t/int64_t/float data type on current device!");
  static_assert(std::is_same_v<pattern, Pattern::Reduce::AR> || std::is_same_v<pattern, Pattern::Reduce::RA>,
                "ReduceProd only support AR and RA pattern on current device!");

  __ubuf__ T *dstAddr = (__ubuf__ T *)dst.GetPhyAddr();
  __ubuf__ T *srcAddr = (__ubuf__ T *)src.GetPhyAddr();
  LocalTensor<T> tmpBuf = sharedTmpBuffer.ReinterpretCast<T>();
  __ubuf__ T *tmpAddr = (__ubuf__ T *)tmpBuf.GetPhyAddr();
  if constexpr (std::is_same_v<pattern, Pattern::Reduce::AR>) {
    if constexpr (SupportBytes<T, 8>()) {
      ReduceProdARB64ReuseSourceCompute<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape);
    } else {
      ReduceProdARImpl<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[0], srcShape[1]);
    }
  } else {
    if constexpr (SupportBytes<T, 8>()) {
      ReduceProdRAB64ReuseSourceCompute<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape);
    } else {
      ReduceProdRAReuseSourceCompute<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape);
    }
  }
}
}  // namespace Internal
}  // namespace ReduceExtendInternal
}  // namespace AscendC
#endif  // AUTOFUSE_REDUCE_EXTEND_REDUCE_PROD_REDUCE_PROD_C310_IMPL_H_
