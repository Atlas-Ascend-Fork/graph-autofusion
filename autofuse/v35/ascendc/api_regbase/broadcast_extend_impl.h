/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_PAD_BROADCAST_BROADCAST_EXTEND_IMPL_H
#define IMPL_PAD_BROADCAST_BROADCAST_EXTEND_IMPL_H

#include "kernel_basic_intf.h"
#include "kernel_tensor.h"
#include "broadcast_extend_utils.h"
#include "broadcast_3510_extend_impl.h"

namespace AscendC {

struct BroadcastTilingExtend {
  uint32_t oriRank;
  uint32_t rank;
  uint32_t dstSize;
  uint32_t srcSize;
  uint32_t loopNum = 0;
  uint32_t oriSrcShape[9];
  uint32_t oriDstShape[9];
  uint32_t dstShape[9];
  uint32_t dstStride[9];
  uint32_t srcStride[10];
};

template <typename T, int constRank = -1, uint32_t *constDstShape = nullptr, uint32_t *constSrcShape = nullptr>
__aicore__ inline void GetBroadcastTilingInfoExtendImpl(uint32_t rank, const uint32_t *dstShape,
                                                        const uint32_t *srcShape, bool srcInnerPad,
                                                        BroadcastTilingExtend &tiling) {
  static_assert((constRank == -1) || (constRank <= 9 && constRank > 0),
                "constRank only supports -1 and the range between 1 and 9");
  ASCENDC_ASSERT((rank <= 9 && rank > 0), { KERNEL_LOG(KERNEL_ERROR, "rank should be in range [1, 9]"); });
  ASCENDC_ASSERT((constRank == -1) || (constRank == rank),
                 { KERNEL_LOG(KERNEL_ERROR, "constRank should be equal to rank when constRank != -1"); });
  constexpr uint32_t maxDim = 9;

  uint32_t srcSize = 1;
  uint32_t dstSize = 1;
  uint32_t newSrcShape[9];
  for (uint32_t i = 0; i < rank; i++) {
    tiling.dstShape[i] = dstShape[i];
    newSrcShape[i] = srcShape[i];
    srcSize *= srcShape[i];
    dstSize *= dstShape[i];

    tiling.oriSrcShape[i] = srcShape[i];
    tiling.oriDstShape[i] = dstShape[i];
  }
  tiling.oriRank = rank;

  if constexpr (constRank == -1 || constRank > 4) {
    uint32_t begin = 0;
    uint32_t i = 0;
    uint32_t count = 0;
    uint32_t size = 1;
    while (i < tiling.oriRank) {
      while (i < tiling.oriRank && (newSrcShape[i] == 1 && tiling.dstShape[i] != 1)) {
        size *= tiling.dstShape[i++];
      }
      if (i - begin >= 1) {
        tiling.dstShape[count] = size;
        newSrcShape[count] = 1;
        rank -= (i - begin - 1);
        ++count;
      }
      begin = i;
      size = 1;
      while (i < tiling.oriRank && newSrcShape[i] == tiling.dstShape[i]) {
        size *= tiling.dstShape[i++];
      }
      if (i - begin >= 1) {
        tiling.dstShape[count] = size;
        newSrcShape[count] = size;
        rank -= (i - begin - 1);
        ++count;
      }
      begin = i;
      size = 1;
    }
    while (i < maxDim) {
      tiling.dstShape[i] = 1;
      newSrcShape[i] = 1;
      ++i;
    }
  }

  if (sizeof(T) == sizeof(uint64_t) && (srcSize != dstSize)) {
    if (newSrcShape[rank - 1] == 1 && tiling.dstShape[rank - 1] != 1) {
      if (rank < maxDim) {
        tiling.dstShape[rank] = 2;
        newSrcShape[rank] = 2;
        rank += 1;
      } else {
        tiling.loopNum = tiling.dstShape[0];
      }
    } else {
      newSrcShape[rank - 1] *= 2;
      tiling.dstShape[rank - 1] *= 2;
    }
    srcSize *= 2;
    dstSize *= 2;
  }
  tiling.rank = rank;
  tiling.dstSize = dstSize;
  tiling.srcSize = srcSize;
  bool srcStrideZero = false;
  if (tiling.loopNum != 0) {
    if (newSrcShape[0] == 1 && tiling.dstShape[0] != 1) {
      tiling.srcStride[9] = 0;
      srcStrideZero = true;
    }
    for (uint32_t i = 0; i < maxDim - 1; i++) {
      tiling.dstShape[i] = tiling.dstShape[i + 1];
      newSrcShape[i] = newSrcShape[i + 1];
    }
    tiling.dstShape[maxDim - 1] = 2;
    newSrcShape[maxDim - 1] = 2;
  }

  uint32_t lastSrcStride = 1;
  uint32_t lastDstStride = 1;
  int32_t end = rank > maxDim ? maxDim : rank;
  for (int32_t i = end - 1; i >= 0; i--) {
    tiling.dstStride[i] = lastDstStride;
    lastDstStride *= tiling.dstShape[i];
    if (newSrcShape[i] == 1 && tiling.dstShape[i] != 1) {
      tiling.srcStride[i] = 0;
    } else {
      tiling.srcStride[i] = lastSrcStride;
      lastSrcStride *= newSrcShape[i];
    }
  }
  if (tiling.loopNum != 0 && !srcStrideZero) {
    tiling.srcStride[9] = lastSrcStride;
  }
}

template <typename T, int32_t constRank>
__aicore__ inline void BroadcastComputeExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                              const uint32_t *dstShape, const uint32_t *dstStride,
                                              const uint32_t *srcStride, int32_t dim, uint32_t srcSize,
                                              uint32_t dstSize, uint32_t loopNum, const uint32_t *oriSrcShape) {
  BroadcastInternal::DstShapeCheck(dstShape, dim);
  using BrcType = typename AscendC::BroadcastInternal::ExtractUnsignedTypeBySize<sizeof(T)>::T;
  __ubuf__ BrcType *dstUb = (__ubuf__ BrcType *)dst.GetPhyAddr();
  __ubuf__ BrcType *srcUb = (__ubuf__ BrcType *)src.GetPhyAddr();
  bool isReduceBranch = false;
  if (srcSize == dstSize) {
    const uint32_t alignSize = ONE_BLK_SIZE / sizeof(T);
    DataCopy(dst, src, AlignUp(dstSize, alignSize));
  } else if (srcSize == 1) {
    BroadcastInternal::BrcDuplicate<BrcType>(dstUb, srcUb, dstSize);
  } else {
    if (srcStride[dim - 1] == 0) {
      if constexpr (constRank == 2) {
        BroadcastInternal::BrcLastWrapperForTwoDim<BrcType, constRank>(dstUb, srcUb, dstShape);
      } else if constexpr (constRank == 3) {
        BroadcastInternal::BrcLastWrapperForThreeDim<BrcType, constRank>(dstUb, srcUb, dstShape, srcStride);
      } else if constexpr (constRank == 4) {
        if (dstShape[0] == 1 && oriSrcShape[0] == 1) {
          BroadcastInternal::BrcLastWrapperForThreeDim<BrcType, constRank>(dstUb, srcUb, dstShape + 1, srcStride + 1);
        } else {
          BroadcastInternal::BrcLastWrapperForFourDim<BrcType, constRank>(dstUb, srcUb, dstShape, srcStride);
        }
      } else {
        if (dim == 2) {
          isReduceBranch = BroadcastInternal::BrcLastWrapperForTwoDim<BrcType>(dstUb, srcUb, dstShape);
        } else if (dim == 3) {
          isReduceBranch = BroadcastInternal::BrcLastWrapperForThreeDim<BrcType>(dstUb, srcUb, dstShape, srcStride);
        } else if (dim == 4) {
          isReduceBranch = BroadcastInternal::BrcLastWrapperForFourDim<BrcType>(dstUb, srcUb, dstShape, srcStride);
        } else if (dim > 4) {
          isReduceBranch = true;
        }
      }
    } else {
      if constexpr (constRank == 2) {
        if constexpr (sizeof(T) == sizeof(uint64_t)) {
          if (dim != constRank) {
            BroadcastInternal::BrcNlastWrapperForThreeDim<BrcType, constRank>(dstUb, srcUb, dstShape, srcStride);
          } else {
            BroadcastInternal::BrcNlastWrapperForTwoDim<BrcType, constRank>(dstUb, srcUb, dstShape);
          }
        } else {
          BroadcastInternal::BrcNlastWrapperForTwoDim<BrcType, constRank>(dstUb, srcUb, dstShape);
        }
      } else if constexpr (constRank == 3) {
        if constexpr (sizeof(T) == sizeof(uint64_t)) {
          if (dim != constRank) {
            BroadcastInternal::BrcNlastWrapperForFourDim<BrcType, constRank>(dstUb, srcUb, dstShape, srcStride);
          } else {
            BroadcastInternal::BrcNlastWrapperForThreeDim<BrcType, constRank>(dstUb, srcUb, dstShape, srcStride);
          }
        } else {
          BroadcastInternal::BrcNlastWrapperForThreeDim<BrcType, constRank>(dstUb, srcUb, dstShape, srcStride);
        }
      } else if constexpr (constRank == 4) {
        if constexpr (sizeof(T) == sizeof(uint64_t)) {
          if (dim != constRank) {
            BroadcastInternal::BrcNlastWrapperForMoreDim<BrcType>(dstUb, srcUb, dstShape, dstStride, srcStride);
          } else {
            BroadcastInternal::BrcNlastWrapperForFourDim<BrcType, constRank>(dstUb, srcUb, dstShape, srcStride);
          }
        } else {
          if (dstShape[0] == 1 && oriSrcShape[0] == 1) {
            BroadcastInternal::BrcNlastWrapperForThreeDim<BrcType, constRank>(dstUb, srcUb, dstShape + 1,
                                                                              srcStride + 1);
          } else {
            BroadcastInternal::BrcNlastWrapperForFourDim<BrcType, constRank>(dstUb, srcUb, dstShape, srcStride);
          }
        }
      } else {
        if (dim == 2) {
          isReduceBranch = BroadcastInternal::BrcNlastWrapperForTwoDim<BrcType>(dstUb, srcUb, dstShape);
        } else if (dim == 3) {
          isReduceBranch = BroadcastInternal::BrcNlastWrapperForThreeDim<BrcType>(dstUb, srcUb, dstShape, srcStride);
        } else if (dim == 4) {
          isReduceBranch = BroadcastInternal::BrcNlastWrapperForFourDim<BrcType>(dstUb, srcUb, dstShape, srcStride);
        } else if (dim > 4) {
          isReduceBranch = true;
        }
      }
    }
    if (isReduceBranch) {
      loopNum = loopNum == 0 ? 1 : loopNum;
      __ubuf__ BrcType *dstUbTmp = dstUb;
      __ubuf__ BrcType *srcUbTmp = srcUb;
      for (uint16_t h = 0; h < loopNum; ++h) {
        dstUb = dstUbTmp + h * dstStride[0] * dstShape[0];
        srcUb = srcUbTmp + h * srcStride[9];
        if (srcStride[dim - 1] == 0) {
          BroadcastInternal::BrcLastWrapperForMoreDimDynamicShape<BrcType>(dstUb, srcUb, dim, dstShape, dstStride,
                                                                           srcStride);
        } else {
          BroadcastInternal::BrcNlastWrapperForMoreDimDynamicShape<BrcType>(dstUb, srcUb, dim, dstShape, dstStride,
                                                                            srcStride);
        }
      }
    }
  }
}

template <typename T, int constRank = -1, uint32_t *constDstShape = nullptr, uint32_t *constSrcShape = nullptr,
          bool constSrcInnerPad = false>
__aicore__ inline void BroadcastExtendImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                           const uint32_t *dstShape, const uint32_t *srcShape,
                                           BroadcastTilingExtend *tiling) {
  CheckTensorPos<T>(dst, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "Broadcast");
  CheckTensorPos<T>(src, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "Broadcast");
  static_assert((constRank == -1) || (constRank <= 9 && constRank > 0),
                "constRank only supports -1 and the range between 1 and 9");
  static_assert(SupportBytes<T, 1, 2, 4, 8>(), "Broadcast only supports type b8/b16/b32/b64 on current device");
  ASCENDC_ASSERT((tiling != nullptr), "BroadcastTilingExtend could not be empty!");
  if constexpr (constRank != -1) {
    ASCENDC_ASSERT((tiling->oriRank == constRank),
                   { KERNEL_LOG(KERNEL_ERROR, "Tiling original rank and constRank should be equal!"); });
  }
  BroadcastInternal::ShapeCheck(tiling->oriDstShape, dstShape, tiling->oriRank);
  BroadcastInternal::ShapeCheck(tiling->oriSrcShape, srcShape, tiling->oriRank);

  uint32_t srcSize = tiling->srcSize;
  uint32_t dstSize = tiling->dstSize;
  uint32_t loopNum = tiling->loopNum;
  uint32_t *dstStride = tiling->dstStride;
  uint32_t *srcStride = tiling->srcStride;
  uint32_t *newDstShape = tiling->dstShape;
  int32_t dim = tiling->rank;

  BroadcastComputeExtend<T, constRank>(dst, src, newDstShape, dstStride, srcStride, dim, srcSize, dstSize, loopNum,
                                       tiling->oriSrcShape);
}

}  // namespace AscendC
#endif  // IMPL_PAD_BROADCAST_BROADCAST_EXTEND_IMPL_H
