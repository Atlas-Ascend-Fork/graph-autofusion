/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ARGMAX_H__
#define __ASCENDC_API_REGBASE_ARGMAX_H__
#ifndef __ASCENDC_API_REGBASE_ARG_REDUCE_H__
#include "arg_reduce.h"
#endif

namespace AscendC {
template <typename U, class pattern>
__aicore__ inline void ArgMaxExtend(const LocalTensor<int32_t> &dst, const LocalTensor<U> &src,
                                    const uint32_t *srcShape, bool srcInnerPad, uint32_t validR = 0U,
                                    uint32_t validA = 0U) {
  if ASCEND_IS_AIC {
    return;
  }
  ArgReduceInput<U, pattern, false, true>(dst, src, srcShape, 0U, validR, validA);
}

template <typename U, class pattern>
__aicore__ inline void ArgMaxWithValueExtend(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &valueDst,
                                             const LocalTensor<U> &src, const LocalTensor<uint8_t> &sharedTmpBuffer,
                                             const uint32_t *srcShape, bool srcInnerPad, uint32_t indexBase = 0U,
                                             uint32_t validR = 0U, uint32_t validA = 0U) {
  if ASCEND_IS_AIC {
    return;
  }
  ArgReduceInput<U, pattern, true, true>(indexDst, src, srcShape, indexBase, validR, validA,
                                         (__ubuf__ U *)valueDst.GetPhyAddr());
}

template <typename U, class pattern>
__aicore__ inline void ArgMaxWithValueExtend(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &valueDst,
                                             const LocalTensor<U> &src, const uint32_t *srcShape, bool srcInnerPad,
                                             uint32_t indexBase = 0U, uint32_t validR = 0U, uint32_t validA = 0U) {
  ArgMaxWithValueExtend<U, pattern>(indexDst, valueDst, src, LocalTensor<uint8_t>(), srcShape, srcInnerPad, indexBase,
                                    validR, validA);
}

template <typename U, class pattern>
__aicore__ inline void ArgMaxWithValueAccumulateExtend(const LocalTensor<int32_t> &indexDst,
                                                       const LocalTensor<U> &valueDst, const LocalTensor<U> &src,
                                                       const LocalTensor<int32_t> &temporaryIndex,
                                                       const LocalTensor<U> &temporaryValue, const uint32_t *srcShape,
                                                       uint32_t indexBase, uint32_t validR, bool firstTile,
                                                       uint32_t validA = 0U) {
  if ASCEND_IS_AIC {
    return;
  }
  ArgReduceAccumulate<U, pattern, true>(indexDst, valueDst, src, temporaryIndex, temporaryValue, srcShape, indexBase,
                                        validR, firstTile, validA);
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2Extend(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &valueSrc,
                                                const LocalTensor<int32_t> &indexSrc,
                                                const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t *srcShape,
                                                uint32_t validA = 0U) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(SupportType<U, float, int32_t>(), "ArgMaxMultiRPhase2Extend: unsupported value type");
  ArgReducePhase2<U, false, true>(indexDst, valueSrc, indexSrc, srcShape, validA);
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2Extend(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &valueSrc,
                                                const LocalTensor<int32_t> &indexSrc, const uint32_t *srcShape,
                                                uint32_t validA = 0U) {
  ArgMaxMultiRPhase2Extend<U>(indexDst, valueSrc, indexSrc, LocalTensor<uint8_t>(), srcShape, validA);
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2VectorExtend(const LocalTensor<int32_t> &indexDst,
                                                      const LocalTensor<U> &valueSrc,
                                                      const LocalTensor<int32_t> &indexSrc,
                                                      const LocalTensor<uint8_t> &sharedTmpBuffer,
                                                      const uint32_t *srcShape, uint32_t validA = 0U) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(SupportType<U, float, int32_t, int64_t>(), "Unsupported Phase2 value type");
  ArgReducePhase2<U, false, true>(indexDst, valueSrc, indexSrc, srcShape, validA);
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2VectorExtend(const LocalTensor<int32_t> &indexDst,
                                                      const LocalTensor<U> &valueSrc,
                                                      const LocalTensor<int32_t> &indexSrc, const uint32_t *srcShape,
                                                      uint32_t validA = 0U) {
  ArgMaxMultiRPhase2VectorExtend<U>(indexDst, valueSrc, indexSrc, LocalTensor<uint8_t>(), srcShape, validA);
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2WithValueVectorExtend(const LocalTensor<U> &valueDst,
                                                               const LocalTensor<int32_t> &indexDst,
                                                               const LocalTensor<U> &valueSrc,
                                                               const LocalTensor<int32_t> &indexSrc,
                                                               const LocalTensor<uint8_t> &sharedTmpBuffer,
                                                               const uint32_t *srcShape, uint32_t validA = 0U) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(SupportType<U, float, int32_t, int64_t>(), "Unsupported Phase2 value type");
  ArgReducePhase2<U, true, true>(indexDst, valueSrc, indexSrc, srcShape, validA, (__ubuf__ U *)valueDst.GetPhyAddr());
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2WithValueVectorExtend(const LocalTensor<U> &valueDst,
                                                               const LocalTensor<int32_t> &indexDst,
                                                               const LocalTensor<U> &valueSrc,
                                                               const LocalTensor<int32_t> &indexSrc,
                                                               const uint32_t *srcShape, uint32_t validA = 0U) {
  ArgMaxMultiRPhase2WithValueVectorExtend<U>(valueDst, indexDst, valueSrc, indexSrc, LocalTensor<uint8_t>(), srcShape,
                                             validA);
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2WithValueExtend(const LocalTensor<U> &valueDst,
                                                         const LocalTensor<int32_t> &indexDst,
                                                         const LocalTensor<U> &valueSrc,
                                                         const LocalTensor<int32_t> &indexSrc,
                                                         const LocalTensor<uint8_t> &sharedTmpBuffer,
                                                         const uint32_t *srcShape, uint32_t validA = 0U) {
  ArgMaxMultiRPhase2WithValueVectorExtend<U>(valueDst, indexDst, valueSrc, indexSrc, sharedTmpBuffer, srcShape, validA);
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2WithValueExtend(const LocalTensor<U> &valueDst,
                                                         const LocalTensor<int32_t> &indexDst,
                                                         const LocalTensor<U> &valueSrc,
                                                         const LocalTensor<int32_t> &indexSrc, const uint32_t *srcShape,
                                                         uint32_t validA = 0U) {
  ArgMaxMultiRPhase2WithValueExtend<U>(valueDst, indexDst, valueSrc, indexSrc, LocalTensor<uint8_t>(), srcShape,
                                       validA);
}

// Compatibility helpers for existing callers; copy does not depend on extremum direction.
template <typename T>
__aicore__ inline void ArgmaxCopy(const LocalTensor<T> &dst, const LocalTensor<T> &src, uint32_t count) {
  ArgReduceCopy(dst, src, count);
}

template <typename U>
__aicore__ inline void ArgmaxMergeRound(const LocalTensor<U> &bestValue, const LocalTensor<int32_t> &bestIndex,
                                        const LocalTensor<U> &curValue, const LocalTensor<int32_t> &curIndex,
                                        uint32_t dimA) {
  ArgReduceMergeRound<U, true>(bestValue, bestIndex, curValue, curIndex, dimA);
}
}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_ARGMAX_H__
