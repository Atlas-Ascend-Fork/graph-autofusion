/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_V35_ASCENDC_API_REGBASE_REDUCE_EXTEND_H
#define AUTOFUSE_V35_ASCENDC_API_REGBASE_REDUCE_EXTEND_H

#include "adv_api/reduce/reduce.h"

#include "reduce/reduce_common_util_3510_impl.h"
#include "reduce/reduce_common_ar_reuse_align_less_than_vl_3510_impl.h"
#include "reduce/reduce_common_ar_reuse_align_3510_impl.h"
#include "reduce/reduce_common_ra_reuse_align_3510_impl.h"
#include "reduce/reduce_common_ar_ra_reuse_unalign_3510_impl.h"
#include "reduce/reduce_sum_3510_impl.h"
#include "reduce/reduce_mean_3510_impl.h"
#include "reduce/reduce_max_3510_impl.h"
#include "reduce/reduce_min_3510_impl.h"
#include "reduce/reduce_prod_3510_impl.h"
#include "reduce/reduce_any_3510_impl.h"
#include "reduce/reduce_all_3510_impl.h"
#include "reduce/mean_3510_impl.h"
#include "reduce/sum_3510_impl.h"
#include "reduce/reduce_xor_sum_3510_impl.h"

namespace AscendC {
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceSumExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                       const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t srcShape[],
                                       bool srcInnerPad) {
  if ASCEND_IS_AIC {
    return;
  }
  ReduceExtendInternal::Internal::ReduceSumImpl<T, pattern, isReuseSource>(dst, src, sharedTmpBuffer, srcShape,
                                                                           srcInnerPad);
}

template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceSumExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t srcShape[],
                                       bool srcInnerPad) {
  LocalTensor<uint8_t> sharedTmpBuffer;
  bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
  ASCENDC_ASSERT((ans), {
    KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer failed to allocate tmpBuffer for ReduceSumExtend, ret=%d",
               static_cast<int32_t>(ans));
  });
  ReduceSumExtend<T, pattern, isReuseSource>(dst, src, sharedTmpBuffer, srcShape, srcInnerPad);
}

template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMeanExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                        const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t srcShape[],
                                        bool srcInnerPad) {
  if ASCEND_IS_AIC {
    return;
  }
  ReduceExtendInternal::Internal::ReduceMeanImpl<T, pattern, isReuseSource>(dst, src, sharedTmpBuffer, srcShape,
                                                                            srcInnerPad);
}

template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMeanExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t srcShape[],
                                        bool srcInnerPad) {
  LocalTensor<uint8_t> sharedTmpBuffer;
  bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
  ASCENDC_ASSERT((ans), {
    KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer failed to allocate tmpBuffer for ReduceMeanExtend, ret=%d",
               static_cast<int32_t>(ans));
  });
  ReduceMeanExtend<T, pattern, isReuseSource>(dst, src, sharedTmpBuffer, srcShape, srcInnerPad);
}

#define AUTOFUSE_DEFINE_REDUCE_EXTEND(apiName, implName)                                                           \
  template <class T, class pattern, bool isReuseSource = false>                                                    \
  __aicore__ inline void apiName(const LocalTensor<T> &dst, const LocalTensor<T> &src,                             \
                                 const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t srcShape[],           \
                                 bool srcInnerPad) {                                                               \
    if ASCEND_IS_AIC {                                                                                             \
      return;                                                                                                      \
    }                                                                                                              \
    ReduceExtendInternal::Internal::implName##Impl<T, pattern, isReuseSource>(dst, src, sharedTmpBuffer, srcShape, \
                                                                              srcInnerPad);                        \
  }                                                                                                                \
  template <class T, class pattern, bool isReuseSource = false>                                                    \
  __aicore__ inline void apiName(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t srcShape[],  \
                                 bool srcInnerPad) {                                                               \
    LocalTensor<uint8_t> sharedTmpBuffer;                                                                          \
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);                                           \
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });                                 \
    apiName<T, pattern, isReuseSource>(dst, src, sharedTmpBuffer, srcShape, srcInnerPad);                          \
  }

AUTOFUSE_DEFINE_REDUCE_EXTEND(ReduceMaxExtend, ReduceMax)
AUTOFUSE_DEFINE_REDUCE_EXTEND(ReduceMinExtend, ReduceMin)
AUTOFUSE_DEFINE_REDUCE_EXTEND(ReduceProdExtend, ReduceProd)
AUTOFUSE_DEFINE_REDUCE_EXTEND(ReduceAnyExtend, ReduceAny)
AUTOFUSE_DEFINE_REDUCE_EXTEND(ReduceAllExtend, ReduceAll)

#undef AUTOFUSE_DEFINE_REDUCE_EXTEND

template <typename T, typename accType = T, bool isReuseSource = false, bool isBasicBlock = false,
          int32_t reduceDim = -1>
__aicore__ inline void MeanExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                  const LocalTensor<uint8_t> &sharedTmpBuffer, const MeanParams &meanParams) {
  ReduceExtendInternal::MeanImpl<T, accType, isReuseSource, isBasicBlock, reduceDim>(dst, src, sharedTmpBuffer,
                                                                                     meanParams);
}

template <typename T, typename accType = T, bool isReuseSource = false, bool isBasicBlock = false,
          int32_t reduceDim = -1>
__aicore__ inline void MeanExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src, const MeanParams &meanParams) {
  LocalTensor<uint8_t> sharedTmpBuffer;
  bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
  ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
  MeanExtend<T, accType, isReuseSource, isBasicBlock, reduceDim>(dst, src, sharedTmpBuffer, meanParams);
}

template <typename T, int32_t reduceDim = -1, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SumExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                 const LocalTensor<uint8_t> &sharedTmpBuffer, const SumParams &sumParams) {
  ReduceExtendInternal::SumCompute<T, reduceDim, isReuseSource, isBasicBlock>(dst, src, sharedTmpBuffer, sumParams);
}

template <typename T, int32_t reduceDim = -1, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SumExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src, const SumParams &sumParams) {
  LocalTensor<uint8_t> sharedTmpBuffer;
  bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
  ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
  SumExtend<T, reduceDim, isReuseSource, isBasicBlock>(dst, src, sharedTmpBuffer, sumParams);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void ReduceXorSumExtend(LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<T> &src1,
                                          LocalTensor<uint8_t> &sharedTmpBuffer, uint32_t calCount) {
  ReduceExtendInternal::ReduceXorSumCompute<T, isReuseSource>(dst, src0, src1, sharedTmpBuffer, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void ReduceXorSumExtend(LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<T> &src1,
                                          uint32_t calCount) {
  LocalTensor<uint8_t> sharedTmpBuffer;
  bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
  ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
  ReduceXorSumExtend<T, isReuseSource>(dst, src0, src1, sharedTmpBuffer, calCount);
}
}  // namespace AscendC

#endif  // AUTOFUSE_V35_ASCENDC_API_REGBASE_REDUCE_EXTEND_H
