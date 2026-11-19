/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ARG_REDUCE_H__
#define __ASCENDC_API_REGBASE_ARG_REDUCE_H__

#ifndef __ASCENDC_API_REGBASE_ARG_REDUCE_CORE_H__
#include "arg_reduce_core.h"
#endif
#ifndef __ASCENDC_API_REGBASE_ARG_REDUCE_MERGE_H__
#include "arg_reduce_merge.h"
#endif

namespace AscendC {
// Shape/dispatch is independent of extremum direction. Public facades select it at compile time.
template <typename U, class pattern, bool withValue, bool isMax>
__aicore__ inline void ArgReduceInput(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &src,
                                      const uint32_t *srcShape, uint32_t indexBase, uint32_t validR, uint32_t validA,
                                      __ubuf__ U *valueDst = nullptr) {
  static_assert(SupportType<U, float, int32_t, int64_t>(), "Unsupported arg-reduce value type");
  static_assert(SupportType<pattern, Pattern::Reduce::AR, Pattern::Reduce::RA>(), "Unsupported arg-reduce pattern");
  constexpr bool isAr = SupportType<pattern, Pattern::Reduce::AR>();
  const uint32_t dimA = srcShape[isAr ? 0 : 1];
  const uint32_t dimR = srcShape[isAr ? 1 : 0];
  constexpr uint32_t vlSize = GetVecLen() / sizeof(U);
  const uint32_t actualR = (validR == 0U || validR > dimR) ? dimR : validR;
  const uint32_t actualA = (validA == 0U || validA > dimA) ? dimA : validA;
  __ubuf__ int32_t *dst = (__ubuf__ int32_t *)indexDst.GetPhyAddr();
  __ubuf__ U *input = (__ubuf__ U *)src.GetPhyAddr();
  if constexpr (isAr) {
    if (dimR % (32U / sizeof(U)) == 0U) {
      asc_vf_call<ArgReduceRowImpl<U, false, true, withValue, isMax>>(dst, input, dimA, dimR, indexBase, actualR,
                                                                      valueDst, actualA);
    } else {
      asc_vf_call<ArgReduceRowImpl<U, false, false, withValue, isMax>>(dst, input, dimA, dimR, indexBase, actualR,
                                                                       valueDst, actualA);
    }
  } else {
    if (actualA < dimA) {
      asc_vf_call<ArgReduceRowImpl<U, true, false, withValue, isMax>>(dst, input, dimA, dimR, indexBase, actualR,
                                                                      valueDst, actualA);
      return;
    }
    if (dimA % (4U * vlSize) == 0U) {
      asc_vf_call<ArgReduceRaFullImpl<U, withValue, 4U, false, isMax>>(dst, input, dimA, actualR, indexBase, valueDst,
                                                                       true);
      return;
    }
    if (dimA % vlSize == 0U) {
      asc_vf_call<ArgReduceRaFullImpl<U, withValue, 1U, false, isMax>>(dst, input, dimA, actualR, indexBase, valueDst,
                                                                       true);
      return;
    }
    if (dimA >= vlSize || (actualR <= vlSize && actualR <= 2U * dimA)) {
      asc_vf_call<ArgReduceRaImpl<U, withValue, isMax>>(dst, input, dimA, actualR, indexBase, valueDst);
      return;
    }
    asc_vf_call<ArgReduceRowImpl<U, true, false, withValue, isMax>>(dst, input, dimA, dimR, indexBase, actualR,
                                                                    valueDst, actualA);
  }
}

// Full-VL RA fuses the current tile and history in one VF; other layouts keep the existing copy/merge path.
template <typename U, class pattern, bool isMax>
__aicore__ inline void ArgReduceAccumulate(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &valueDst,
                                           const LocalTensor<U> &src, const LocalTensor<int32_t> &temporaryIndex,
                                           const LocalTensor<U> &temporaryValue, const uint32_t *srcShape,
                                           uint32_t indexBase, uint32_t validR, bool firstTile, uint32_t validA) {
  static_assert(SupportType<U, float, int32_t, int64_t>(), "Unsupported arg-reduce value type");
  static_assert(SupportType<pattern, Pattern::Reduce::AR, Pattern::Reduce::RA>(), "Unsupported arg-reduce pattern");
  constexpr bool isAr = SupportType<pattern, Pattern::Reduce::AR>();
  const uint32_t dimA = srcShape[isAr ? 0 : 1];
  const uint32_t dimR = srcShape[isAr ? 1 : 0];
  const uint32_t actualR = (validR == 0U || validR > dimR) ? dimR : validR;
  const uint32_t actualA = (validA == 0U || validA > dimA) ? dimA : validA;
  if constexpr (!isAr) {
    if (actualA == dimA && dimA % (4U * (GetVecLen() / sizeof(U))) == 0U) {
      asc_vf_call<ArgReduceRaFullImpl<U, true, 4U, true, isMax>>(
          (__ubuf__ int32_t *)indexDst.GetPhyAddr(), (__ubuf__ U *)src.GetPhyAddr(), dimA, actualR, indexBase,
          (__ubuf__ U *)valueDst.GetPhyAddr(), firstTile);
      return;
    }
    if (actualA == dimA && dimA % (GetVecLen() / sizeof(U)) == 0U) {
      asc_vf_call<ArgReduceRaFullImpl<U, true, 1U, true, isMax>>(
          (__ubuf__ int32_t *)indexDst.GetPhyAddr(), (__ubuf__ U *)src.GetPhyAddr(), dimA, actualR, indexBase,
          (__ubuf__ U *)valueDst.GetPhyAddr(), firstTile);
      return;
    }
  }
  ArgReduceInput<U, pattern, true, isMax>(temporaryIndex, src, srcShape, indexBase, validR, validA,
                                          (__ubuf__ U *)temporaryValue.GetPhyAddr());
  if (firstTile) {
    ArgReduceCopy(indexDst, temporaryIndex, actualA);
    ArgReduceCopy(valueDst, temporaryValue, actualA);
  } else {
    ArgReduceMergeRound<U, isMax>(valueDst, indexDst, temporaryValue, temporaryIndex, actualA);
  }
}

template <typename U, bool withValue, bool isMax>
__aicore__ inline void ArgReducePhase2(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &valueSrc,
                                       const LocalTensor<int32_t> &indexSrc, const uint32_t *srcShape, uint32_t validA,
                                       __ubuf__ U *valueDst = nullptr) {
  const uint32_t dimN = srcShape[0];
  const uint32_t dimAValue = srcShape[1];
  const uint32_t dimAIndex = srcShape[2];
  const uint32_t maxA = dimAValue < dimAIndex ? dimAValue : dimAIndex;
  const uint32_t actualA = (validA == 0U || validA > maxA) ? maxA : validA;
  asc_vf_call<ArgReducePhase2VectorImpl<U, withValue, isMax>>(
      (__ubuf__ int32_t *)indexDst.GetPhyAddr(), (__ubuf__ U *)valueSrc.GetPhyAddr(),
      (__ubuf__ int32_t *)indexSrc.GetPhyAddr(), dimN, dimAValue, dimAIndex, actualA, valueDst);
}
}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_ARG_REDUCE_H__
