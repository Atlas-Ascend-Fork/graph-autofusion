/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_REDUCE_EXTEND_REDUCE_ALL_REDUCE_ALL_C310_IMPL_H_
#define AUTOFUSE_REDUCE_EXTEND_REDUCE_ALL_REDUCE_ALL_C310_IMPL_H_

#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
#include "reduce/reduce_common_util_3510_impl.h"
#include "reduce/reduce_common_ar_reuse_align_3510_impl.h"
#include "reduce/reduce_common_ar_ra_reuse_unalign_3510_impl.h"
#include "reduce/reduce_prod_3510_impl.h"
#include "reduce/reduce_common_ra_reuse_align_3510_impl.h"

namespace AscendC {
namespace ReduceExtendInternal {
namespace Internal {

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceAllARImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr, uint32_t dimA,
                                       uint32_t dimR) {
  constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
  using CastType = typename ReduceOpInternal::template ExtractReduceCastType<T>::CastT;

  if (dimR <= vlSize) {
    ReduceARReuseSourceLessThanVL<T, Reg::RegTraitNumOne, vlSize,
                                  Reg::Min<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
                                  Reg::ReduceMin<CastType, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<CastType>>>(
        dstAddr, srcAddr, dimA, dimR);
  } else {
    ReduceAROverVLImpl<T, Reg::RegTraitNumOne, vlSize, Reg::Min<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>,
                       Reg::ReduceMin<CastType, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<CastType>>, isReuseSource>(
        dstAddr, srcAddr, tmpAddr, dimA, dimR);
  }
}

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceAllB64ARCompute(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
                                             const uint32_t srcShape[]) {
  if ((srcShape[1] * sizeof(T)) % 32 == 0) {
    ReduceARImpl<T, Reg::RegTraitNumTwo,
                 Reg::Min<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
                 Reg::ReduceMin<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>, isReuseSource>(
        dstAddr, srcAddr, tmpAddr, srcShape[0], srcShape[1]);
  } else {
    if constexpr (!isReuseSource) {
      ReduceProdCopyUnaligned<T>(tmpAddr, srcAddr, srcShape[0] * srcShape[1]);
      srcAddr = tmpAddr;
    }
    ReduceARReuseSourceUnAligned<
        T, Reg::RegTraitNumTwo, Reg::Min<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
        Reg::ReduceMin<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>>(
        dstAddr, srcAddr, srcShape[0], srcShape[1]);
  }
}

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceAllB64RACompute(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
                                             const uint32_t srcShape[]) {
  if ((srcShape[1] * sizeof(T)) % 32 == 0) {
    ReduceRAB64ReuseSource<T, Reg::RegTraitNumTwo,
                           Reg::Min<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>,
                           isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[1], srcShape[0]);
  } else {
    if constexpr (!isReuseSource) {
      ReduceProdCopyUnaligned<T>(tmpAddr, srcAddr, srcShape[1] * srcShape[0]);
      srcAddr = tmpAddr;
    }
    ReduceRAReuseSourceUnAlignedB64<T, Reg::RegTraitNumTwo,
                                    Reg::Min<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T, Reg::RegTraitNumTwo>>>(
        dstAddr, srcAddr, srcShape[1], srcShape[0]);
  }
}

template <typename T, typename pattern, bool isReuseSource = false>
__aicore__ inline void ReduceAllImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                     const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t srcShape[],
                                     bool srcInnerPad) {
  CHECK_FUNC_HIGHLEVEL_API(ReduceAll, (T, pattern), (dst, src, sharedTmpBuffer, srcShape, srcInnerPad, srcShape[1]));

  CheckTensorPos<T>(dst, Hardware::UB, "dst", "VECIN/VECCALC/VECOUT", "ReduceAll");
  CheckTensorPos<T>(src, Hardware::UB, "src", "VECIN/VECCALC/VECOUT", "ReduceAll");
  CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN/VECCALC/VECOUT", "ReduceAll");
  static_assert(SupportType<T, uint8_t, int8_t, int16_t, int32_t, float, int64_t>(),
                "ReduceAll only support uint8_t/int8_t/int16_t/int32_t/float/int64_t data type on current device!");
  static_assert(std::is_same_v<pattern, Pattern::Reduce::AR> || std::is_same_v<pattern, Pattern::Reduce::RA>,
                "ReduceAll only support AR and RA pattern on current device!");

  __ubuf__ T *dstAddr = (__ubuf__ T *)dst.GetPhyAddr();
  __ubuf__ T *srcAddr = (__ubuf__ T *)src.GetPhyAddr();
  LocalTensor<T> tmpBuf = sharedTmpBuffer.ReinterpretCast<T>();
  __ubuf__ T *tmpAddr = (__ubuf__ T *)tmpBuf.GetPhyAddr();
  if constexpr (std::is_same_v<pattern, Pattern::Reduce::AR>) {
    if constexpr (SupportBytes<T, 8>()) {
      ReduceAllB64ARCompute<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape);
    } else {
      ReduceAllARImpl<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[0], srcShape[1]);
    }
  } else {
    if constexpr (SupportBytes<T, 8>()) {
      ReduceAllB64RACompute<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape);
    } else {
      ReduceRAImpl<T, Reg::RegTraitNumOne, Reg::Min<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>, isReuseSource>(
          dstAddr, srcAddr, tmpAddr, srcShape[1], srcShape[0]);
    }
  }
}
}  // namespace Internal
}  // namespace ReduceExtendInternal
}  // namespace AscendC
#endif  // AUTOFUSE_REDUCE_EXTEND_REDUCE_All_REDUCE_All_C310_IMPL_H_
