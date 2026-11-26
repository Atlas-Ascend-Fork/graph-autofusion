/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_SOFTMAX_H__
#define __ASCENDC_API_REGBASE_SOFTMAX_H__

using namespace AscendC;

namespace SoftmaxRegBase {
constexpr uint32_t BLOCK_SIZE = 32U;
constexpr int64_t CONST_ZERO = 0;
constexpr int64_t CONST_ONE = 1;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t CONST_SIXTY_THREE = 63;
constexpr float NEG_INFINITY = -(__builtin_inff());
constexpr uint32_t VL_FP32 = AscendC::ONE_REPEAT_BYTE_SIZE / sizeof(float);
constexpr uint32_t AR_SMALL_R_THRESHOLD = 8U;
constexpr uint32_t AR_MID_R_THRESHOLD = VL_FP32;

enum class SoftmaxARPathType : uint32_t {
  SMALL_R = 0U,
  MID_R = 1U,
  FULL_LOAD = 2U,
};

__aicore__ inline SoftmaxARPathType GetSoftmaxARPathType(uint32_t rSize) {
  if (rSize <= AR_SMALL_R_THRESHOLD) {
    return SoftmaxARPathType::SMALL_R;
  }
  if (rSize <= AR_MID_R_THRESHOLD) {
    return SoftmaxARPathType::MID_R;
  }
  return SoftmaxARPathType::FULL_LOAD;
}

constexpr static AscendC::Reg::CastTrait cast_trait_fp16_to_fp32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::Reg::CastTrait cast_trait_fp32_to_fp16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t factor) {
  if (factor == 0U) {
    return 0U;
  }
  return (value + factor - 1U) / factor;
}

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t factor) {
  return CeilDiv(value, factor) * factor;
}

__aicore__ inline uint32_t FindNearestPower2(uint32_t value) {
  if (value <= CONST_ONE) {
    return CONST_ZERO;
  } else if (value <= CONST_TWO) {
    return CONST_ONE;
  } else if (value <= CONST_FOUR) {
    return CONST_TWO;
  }
  const uint64_t num = static_cast<uint64_t>(value - CONST_ONE);
  const uint32_t pow = CONST_SIXTY_THREE - AscendC::ScalarCountLeadingZero(num);
  return CONST_ONE << pow;
}

template <typename T>
__aicore__ inline uint32_t CalcRAligned(uint32_t r) {
  return AlignUp(r, BLOCK_SIZE / sizeof(T));
}

__aicore__ inline uint32_t CalcReduceTmpStride(uint32_t r) {
  if (r <= CONST_TWO * VL_FP32) {
    return 0U;
  }
  uint32_t ceilVlCount = CeilDiv(r, VL_FP32);
  uint32_t foldPoint = FindNearestPower2(ceilVlCount);
  return AlignUp(foldPoint, BLOCK_SIZE / sizeof(float));
}

struct SoftmaxARTilingInfo {
  uint32_t aSize;
  uint32_t rSize;
  uint32_t rAligned;
  uint32_t expBufElems;
  uint32_t reduceTmpStride;
  uint32_t reduceTmpElems;
  uint32_t requiredTmpBytes;
};

template <typename TIn>
__aicore__ inline SoftmaxARTilingInfo GetSoftmaxARTilingInfo(uint32_t a, uint32_t r) {
  SoftmaxARTilingInfo tilingInfo{};
  tilingInfo.aSize = a;
  tilingInfo.rSize = r;
  tilingInfo.rAligned = CalcRAligned<TIn>(r);
  tilingInfo.expBufElems = a * tilingInfo.rAligned;
  tilingInfo.reduceTmpStride = CalcReduceTmpStride(r);
  tilingInfo.reduceTmpElems = a * tilingInfo.reduceTmpStride;
  tilingInfo.requiredTmpBytes = (tilingInfo.expBufElems + tilingInfo.reduceTmpElems) * sizeof(float);
  return tilingInfo;
}

__aicore__ inline uint32_t CalcSmallRExpCacheElems(uint32_t r) {
  return r * VL_FP32;
}

__aicore__ inline uint32_t CalcSmallRExpCacheBytes(uint32_t r) {
  return CalcSmallRExpCacheElems(r) * sizeof(float);
}

template <typename TIn>
__aicore__ inline void LoadAsFp32(__local_mem__ TIn *src, AscendC::Reg::RegTensor<float> &dst,
                                  AscendC::Reg::MaskReg &preg, uint32_t offset) {
  if constexpr (IsSameType<TIn, float>::value) {
    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(dst, src + offset);
  } else {
    AscendC::MicroAPI::RegTensor<TIn> srcReg;
    AscendC::MicroAPI::DataCopy<TIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, src + offset);
    AscendC::MicroAPI::Cast<float, TIn, cast_trait_fp16_to_fp32>(dst, srcReg, preg);
  }
}

template <typename TIn>
__aicore__ inline void GatherAsFp32(__local_mem__ TIn *src, AscendC::Reg::RegTensor<float> &dst,
                                    AscendC::Reg::RegTensor<uint32_t> &index, AscendC::Reg::MaskReg &preg) {
  if constexpr (IsSameType<TIn, float>::value) {
    AscendC::MicroAPI::DataCopyGather(dst, src, index, preg);
  } else {
    AscendC::MicroAPI::RegTensor<TIn> srcReg;
    AscendC::MicroAPI::DataCopyGather(srcReg, src, index, preg);
    AscendC::MicroAPI::Cast<float, TIn, cast_trait_fp16_to_fp32>(dst, srcReg, preg);
  }
}

template <typename TOut>
__aicore__ inline void StoreFromFp32(__local_mem__ TOut *dst, AscendC::Reg::RegTensor<float> &src,
                                     AscendC::Reg::MaskReg &preg, uint32_t offset) {
  if constexpr (IsSameType<TOut, float>::value) {
    AscendC::MicroAPI::DataCopy<TOut, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst + offset, src, preg);
  } else {
    AscendC::MicroAPI::RegTensor<TOut> dstReg;
    AscendC::MicroAPI::Cast<TOut, float, cast_trait_fp32_to_fp16>(dstReg, src, preg);
    AscendC::MicroAPI::DataCopy<TOut, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, dstReg, preg);
  }
}

template <typename TOut, typename TIndex>
__aicore__ inline void ScatterFromFp32(__local_mem__ TOut *dst, AscendC::Reg::RegTensor<float> &src,
                                       AscendC::Reg::RegTensor<TIndex> &index, AscendC::Reg::MaskReg &preg) {
  if constexpr (IsSameType<TOut, float>::value) {
    AscendC::MicroAPI::DataCopyScatter(dst, src, index, preg);
  } else {
    AscendC::MicroAPI::RegTensor<TOut> dstReg;
    AscendC::MicroAPI::Cast<TOut, float, cast_trait_fp32_to_fp16>(dstReg, src, preg);
    AscendC::MicroAPI::DataCopyScatter(dst, dstReg, index, preg);
  }
}

template <typename TIn>
__aicore__ inline void FirstNormCompute(uint32_t aSize, uint32_t rSize, uint32_t rAligned, __local_mem__ TIn *srcAddr,
                                        __local_mem__ float *expAddr) {
  const uint16_t actualA = static_cast<uint16_t>(aSize);
  const uint16_t rLoopCount = static_cast<uint16_t>(CeilDiv(rSize, VL_FP32));
  const uint16_t mainLoopCount = rLoopCount - 1U;
  uint32_t tailSize = rSize - VL_FP32 * mainLoopCount;

  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<float> maxReg;
    AscendC::MicroAPI::RegTensor<float> srcReg;
    AscendC::MicroAPI::RegTensor<float> rowMaxReg;
    AscendC::MicroAPI::RegTensor<float> brcMaxReg;
    AscendC::MicroAPI::RegTensor<float> subReg;
    AscendC::MicroAPI::RegTensor<float> expReg;
    AscendC::MicroAPI::MaskReg tailMask;
    AscendC::MicroAPI::MaskReg curMask;
    AscendC::MicroAPI::MaskReg fullMask;

    uint32_t fullSize = VL_FP32;
    tailMask = AscendC::MicroAPI::UpdateMask<uint32_t>(tailSize);
    fullMask = AscendC::MicroAPI::UpdateMask<uint32_t>(fullSize);

    for (uint16_t row = 0U; row < actualA; ++row) {
      uint32_t tailOffset = row * rAligned + VL_FP32 * mainLoopCount;
      AscendC::MicroAPI::Duplicate(maxReg, NEG_INFINITY, fullMask);
      LoadAsFp32(srcAddr, srcReg, tailMask, tailOffset);
      AscendC::MicroAPI::Max(srcReg, maxReg, srcReg, tailMask);
      AscendC::MicroAPI::Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(maxReg, srcReg, tailMask);

      for (uint16_t loopIdx = 0U; loopIdx < mainLoopCount; ++loopIdx) {
        uint32_t curOffset = loopIdx * VL_FP32 + row * rAligned;
        LoadAsFp32(srcAddr, srcReg, fullMask, curOffset);
        AscendC::MicroAPI::Max(maxReg, maxReg, srcReg, fullMask);
      }
      AscendC::MicroAPI::ReduceMax(rowMaxReg, maxReg, fullMask);
      AscendC::MicroAPI::Duplicate(brcMaxReg, rowMaxReg, fullMask);

      for (uint16_t loopIdx = 0U; loopIdx < rLoopCount; ++loopIdx) {
        uint32_t offset = loopIdx * VL_FP32 + row * rAligned;
        uint32_t valid = rSize - loopIdx * VL_FP32;
        valid = valid > VL_FP32 ? VL_FP32 : valid;
        curMask = AscendC::MicroAPI::UpdateMask<float>(valid);
        LoadAsFp32(srcAddr, srcReg, curMask, offset);
        AscendC::MicroAPI::Sub(subReg, srcReg, brcMaxReg, curMask);
        AscendC::MicroAPI::Exp(expReg, subReg, curMask);
        AscendC::MicroAPI::DataCopy((__local_mem__ float *)expAddr + offset, expReg, curMask);
      }
    }
  }
}

template <typename TOut>
__aicore__ inline void NormalizeAndStoreRow(__local_mem__ TOut *dst, __local_mem__ float *oriSrc,
                                            AscendC::Reg::RegTensor<float> &brcSumReg,
                                            AscendC::Reg::RegTensor<float> &outReg, AscendC::Reg::MaskReg &oriMask,
                                            uint16_t row, uint16_t rLoopCount, uint32_t oriR, uint32_t oriRAligned) {
  for (uint16_t loopIdx = 0U; loopIdx < rLoopCount; ++loopIdx) {
    uint32_t valid = oriR - loopIdx * VL_FP32;
    valid = valid > VL_FP32 ? VL_FP32 : valid;
    oriMask = AscendC::MicroAPI::UpdateMask<float>(valid);
    uint32_t offset = loopIdx * VL_FP32 + row * oriRAligned;
    AscendC::MicroAPI::DataCopy(outReg, (__local_mem__ float *)oriSrc + offset);
    AscendC::MicroAPI::Div(outReg, outReg, brcSumReg, oriMask);
    StoreFromFp32(dst, outReg, oriMask, offset);
  }
}

template <typename TIn, typename TOut>
__aicore__ inline void SoftmaxARMidRCompute(const AscendC::LocalTensor<TOut> &dst, const AscendC::LocalTensor<TIn> &src,
                                            const AscendC::LocalTensor<float> &expBuf,
                                            const SoftmaxARTilingInfo &tilingInfo) {
  __local_mem__ TOut *dstAddr = (__local_mem__ TOut *)dst.GetPhyAddr();
  __local_mem__ TIn *srcAddr = (__local_mem__ TIn *)src.GetPhyAddr();
  const uint16_t aLoopTimes = static_cast<uint16_t>(tilingInfo.aSize);
  const uint16_t rLoopCount = static_cast<uint16_t>(CeilDiv(tilingInfo.rSize, VL_FP32));
  const uint32_t rAligned = tilingInfo.rAligned;

  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<float> srcReg;
    AscendC::MicroAPI::RegTensor<float> rowMaxReg;
    AscendC::MicroAPI::RegTensor<float> brcMaxReg;
    AscendC::MicroAPI::RegTensor<float> subReg;
    AscendC::MicroAPI::RegTensor<float> expReg;
    AscendC::MicroAPI::RegTensor<float> sumReg;
    AscendC::MicroAPI::RegTensor<float> brcSumReg;
    AscendC::MicroAPI::RegTensor<float> outReg;
    uint32_t maskValue = tilingInfo.rSize;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskValue);
    AscendC::MicroAPI::MaskReg oriMask;

    for (uint16_t row = 0U; row < aLoopTimes; ++row) {
      uint32_t offset = row * rAligned;
      LoadAsFp32(srcAddr, srcReg, mask, offset);
      AscendC::MicroAPI::ReduceMax(rowMaxReg, srcReg, mask);
      AscendC::MicroAPI::Duplicate(brcMaxReg, rowMaxReg, mask);
      AscendC::MicroAPI::Sub(subReg, srcReg, brcMaxReg, mask);
      AscendC::MicroAPI::Exp(expReg, subReg, mask);
      AscendC::MicroAPI::ReduceSum(sumReg, expReg, mask);
      AscendC::MicroAPI::Duplicate(brcSumReg, sumReg, mask);
      AscendC::MicroAPI::Div(expReg, expReg, brcSumReg, mask);
      StoreFromFp32(dstAddr, expReg, mask, offset);
    }
  }
}

template <typename TOut>
__aicore__ inline void SecondNormComputePostSingleBlock(__local_mem__ TOut *dst, __local_mem__ float *src,
                                                        __local_mem__ float *oriSrc, uint16_t loopTimes,
                                                        uint16_t rLoopCount, uint32_t rSize, uint32_t stride,
                                                        uint32_t oriR, uint32_t oriRAligned) {
  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<float> srcReg;
    AscendC::MicroAPI::RegTensor<float> sumReg;
    AscendC::MicroAPI::RegTensor<float> brcSumReg;
    AscendC::MicroAPI::RegTensor<float> outReg;
    AscendC::MicroAPI::MaskReg reduceMask = AscendC::MicroAPI::UpdateMask<float>(rSize);
    AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg oriMask;
    for (uint16_t row = 0U; row < loopTimes; ++row) {
      AscendC::MicroAPI::DataCopy(srcReg, (__local_mem__ float *)src + row * stride);
      AscendC::MicroAPI::ReduceSum(sumReg, srcReg, reduceMask);
      AscendC::MicroAPI::Duplicate(brcSumReg, sumReg, fullMask);
      NormalizeAndStoreRow(dst, oriSrc, brcSumReg, outReg, oriMask, row, rLoopCount, oriR, oriRAligned);
    }
  }
}

template <typename TOut>
__aicore__ inline void SecondNormComputePostTwoBlock(__local_mem__ TOut *dst, __local_mem__ float *src0,
                                                     __local_mem__ float *src1, __local_mem__ float *oriSrc,
                                                     uint16_t loopTimes, uint16_t rLoopCount, uint32_t rSize,
                                                     uint32_t stride, uint32_t oriR, uint32_t oriRAligned) {
  __VEC_SCOPE__ {
    uint32_t tailCount = rSize - VL_FP32;
    AscendC::MicroAPI::RegTensor<float> reg0;
    AscendC::MicroAPI::RegTensor<float> reg1;
    AscendC::MicroAPI::RegTensor<float> addReg;
    AscendC::MicroAPI::RegTensor<float> sumReg;
    AscendC::MicroAPI::RegTensor<float> brcSumReg;
    AscendC::MicroAPI::RegTensor<float> outReg;
    AscendC::MicroAPI::MaskReg tailMask = AscendC::MicroAPI::UpdateMask<float>(tailCount);
    AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg oriMask;
    for (uint16_t row = 0U; row < loopTimes; ++row) {
      AscendC::MicroAPI::DataCopy(reg0, (__local_mem__ float *)src0 + row * stride);
      AscendC::MicroAPI::DataCopy(reg1, (__local_mem__ float *)src1 + row * stride);
      AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(addReg, reg0, reg1, tailMask);
      AscendC::MicroAPI::Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, addReg, tailMask);
      AscendC::MicroAPI::ReduceSum(sumReg, reg0, fullMask);
      AscendC::MicroAPI::Duplicate(brcSumReg, sumReg, fullMask);
      NormalizeAndStoreRow(dst, oriSrc, brcSumReg, outReg, oriMask, row, rLoopCount, oriR, oriRAligned);
    }
  }
}

template <typename TOut>
__aicore__ inline void SecondNormComputePost(const LocalTensor<TOut> &dstTensor, const LocalTensor<float> &srcTensor,
                                             const LocalTensor<float> &oriSrcTensor, uint32_t aSize, uint32_t rSize,
                                             uint32_t stride, uint32_t oriR, uint32_t oriRAligned) {
  if (aSize == 0U || rSize == 0U || rSize > CONST_TWO * VL_FP32) {
    return;
  }

  uint16_t loopTimes = static_cast<uint16_t>(aSize);
  uint16_t rLoopCount = static_cast<uint16_t>(CeilDiv(oriR, VL_FP32));
  __local_mem__ TOut *dst = (__local_mem__ TOut *)dstTensor.GetPhyAddr();
  __local_mem__ float *src = (__local_mem__ float *)srcTensor.GetPhyAddr();
  __local_mem__ float *oriSrc = (__local_mem__ float *)oriSrcTensor.GetPhyAddr();
  if (rSize <= VL_FP32) {
    SecondNormComputePostSingleBlock(dst, src, oriSrc, loopTimes, rLoopCount, rSize, stride, oriR, oriRAligned);
    return;
  }
  SecondNormComputePostTwoBlock(dst, src, src + VL_FP32, oriSrc, loopTimes, rLoopCount, rSize, stride, oriR,
                                oriRAligned);
}

__aicore__ inline void SecondNormMainFoldRow(__local_mem__ float *foldSrcA, __local_mem__ float *foldSrcB,
                                             __local_mem__ float *&dst, AscendC::MicroAPI::UnalignReg &unalignReg,
                                             AscendC::MicroAPI::MaskReg &fullMask, uint16_t row, uint32_t stride,
                                             uint16_t mainFoldLoopTimes) {
  for (uint16_t loopIdx = 0U; loopIdx < mainFoldLoopTimes; ++loopIdx) {
    AscendC::MicroAPI::RegTensor<float> reg0;
    AscendC::MicroAPI::RegTensor<float> reg1;
    AscendC::MicroAPI::RegTensor<float> addReg;
    AscendC::MicroAPI::RegTensor<float> sumReg;
    AscendC::MicroAPI::DataCopy(reg0, (__local_mem__ float *)foldSrcA + row * stride + loopIdx * VL_FP32);
    AscendC::MicroAPI::DataCopy(reg1, (__local_mem__ float *)foldSrcB + row * stride + loopIdx * VL_FP32);
    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(addReg, reg0, reg1, fullMask);
    AscendC::MicroAPI::ReduceSum(sumReg, addReg, fullMask);
    AscendC::MicroAPI::DataCopyUnAlign(dst, sumReg, unalignReg, 1);
  }
}

__aicore__ inline void SecondNormTailFoldRow(__local_mem__ float *tailSrcA, __local_mem__ float *tailSrcB,
                                             __local_mem__ float *&dst, AscendC::MicroAPI::UnalignReg &unalignReg,
                                             AscendC::MicroAPI::MaskReg &fullMask, uint16_t row, uint32_t stride,
                                             uint16_t tailFoldLoopTimes, uint32_t tailFoldElemCount) {
  for (uint16_t loopIdx = 0U; loopIdx < tailFoldLoopTimes; ++loopIdx) {
    AscendC::MicroAPI::RegTensor<float> reg0;
    AscendC::MicroAPI::RegTensor<float> reg1;
    AscendC::MicroAPI::RegTensor<float> addReg;
    uint32_t count = static_cast<uint32_t>(tailFoldElemCount);
    AscendC::MicroAPI::MaskReg tailMask = AscendC::MicroAPI::UpdateMask<float>(count);
    AscendC::MicroAPI::DataCopy(reg0, (__local_mem__ float *)tailSrcA + row * stride + loopIdx * VL_FP32);
    AscendC::MicroAPI::DataCopy(reg1, (__local_mem__ float *)tailSrcB + row * stride + loopIdx * VL_FP32);
    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(addReg, reg0, reg1, tailMask);
    AscendC::MicroAPI::Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, addReg, tailMask);
    AscendC::MicroAPI::ReduceSum(reg1, reg0, fullMask);
    AscendC::MicroAPI::DataCopyUnAlign(dst, reg1, unalignReg, 1);
  }
}

__aicore__ inline void SecondNormUnFoldRow(__local_mem__ float *unFoldSrc, __local_mem__ float *&dst,
                                           AscendC::MicroAPI::UnalignReg &unalignReg,
                                           AscendC::MicroAPI::MaskReg &fullMask, uint16_t row, uint32_t stride,
                                           uint16_t unFoldLoopTimes) {
  for (uint16_t loopIdx = 0U; loopIdx < unFoldLoopTimes; ++loopIdx) {
    AscendC::MicroAPI::RegTensor<float> reg0;
    AscendC::MicroAPI::RegTensor<float> sumReg;
    AscendC::MicroAPI::DataCopy(reg0, (__local_mem__ float *)unFoldSrc + row * stride + loopIdx * VL_FP32);
    AscendC::MicroAPI::ReduceSum(sumReg, reg0, fullMask);
    AscendC::MicroAPI::DataCopyUnAlign(dst, sumReg, unalignReg, 1);
  }
}

template <typename TOut>
__aicore__ inline void SecondNormCompute(const LocalTensor<TOut> &dstTensor, const LocalTensor<float> &srcTensor,
                                         const LocalTensor<float> &reduceSumTempTensor, uint32_t aSize, uint32_t rSize,
                                         uint32_t stride) {
  if (aSize == 0U || rSize == 0U) {
    return;
  }
  if (rSize <= CONST_TWO * VL_FP32) {
    SecondNormComputePost(dstTensor, srcTensor, srcTensor, aSize, rSize, stride, rSize, stride);
    return;
  }

  uint32_t ceilVlCount = CeilDiv(rSize, VL_FP32);
  uint32_t floorVlCount = rSize / VL_FP32;
  uint32_t foldPoint = FindNearestPower2(ceilVlCount);
  uint16_t outerLoopTimes = static_cast<uint16_t>(aSize);
  uint16_t tailFoldLoopTimes = static_cast<uint16_t>(ceilVlCount - floorVlCount);
  uint32_t tailFoldElemCount = rSize - floorVlCount * VL_FP32;
  uint16_t mainFoldLoopTimes = static_cast<uint16_t>(floorVlCount - foldPoint);
  uint16_t unFoldLoopTimes = static_cast<uint16_t>(foldPoint + foldPoint - ceilVlCount);
  uint32_t outerLoopDstStride = AlignUp(foldPoint, BLOCK_SIZE / sizeof(float));
  uint32_t foldSrcBOffset = foldPoint * VL_FP32;
  uint32_t tailSrcAOffset = mainFoldLoopTimes * VL_FP32;
  uint32_t tailSrcBOffset = floorVlCount * VL_FP32;
  uint32_t unFoldSrcOffset = (mainFoldLoopTimes + tailFoldLoopTimes) * VL_FP32;

  __local_mem__ float *foldSrcA = (__local_mem__ float *)srcTensor.GetPhyAddr();
  __local_mem__ float *foldSrcB = (__local_mem__ float *)srcTensor.GetPhyAddr() + foldSrcBOffset;
  __local_mem__ float *tailSrcA = (__local_mem__ float *)srcTensor.GetPhyAddr() + tailSrcAOffset;
  __local_mem__ float *tailSrcB = (__local_mem__ float *)srcTensor.GetPhyAddr() + tailSrcBOffset;
  __local_mem__ float *unFoldSrc = (__local_mem__ float *)srcTensor.GetPhyAddr() + unFoldSrcOffset;

  __VEC_SCOPE__ {
    AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::UnalignReg unalignReg;

    for (uint16_t row = 0U; row < outerLoopTimes; ++row) {
      __local_mem__ float *dst = (__local_mem__ float *)reduceSumTempTensor.GetPhyAddr() + row * outerLoopDstStride;
      SecondNormMainFoldRow(foldSrcA, foldSrcB, dst, unalignReg, fullMask, row, stride, mainFoldLoopTimes);
      SecondNormTailFoldRow(tailSrcA, tailSrcB, dst, unalignReg, fullMask, row, stride, tailFoldLoopTimes,
                            tailFoldElemCount);
      SecondNormUnFoldRow(unFoldSrc, dst, unalignReg, fullMask, row, stride, unFoldLoopTimes);
      AscendC::MicroAPI::DataCopyUnAlignPost(dst, unalignReg, 0);
    }
  }
  SecondNormComputePost(dstTensor, reduceSumTempTensor, srcTensor, aSize, foldPoint, outerLoopDstStride, rSize, stride);
}

template <typename TIn, typename TOut>
__aicore__ inline void SoftmaxARFullLoadCompute(const AscendC::LocalTensor<TOut> &dst,
                                                const AscendC::LocalTensor<TIn> &src,
                                                const AscendC::LocalTensor<float> &expBuf,
                                                const AscendC::LocalTensor<float> &reduceTmpBuf,
                                                const SoftmaxARTilingInfo &tilingInfo) {
  __local_mem__ TIn *srcAddr = (__local_mem__ TIn *)src.GetPhyAddr();
  __local_mem__ float *expAddr = (__local_mem__ float *)expBuf.GetPhyAddr();
  FirstNormCompute<TIn>(tilingInfo.aSize, tilingInfo.rSize, tilingInfo.rAligned, srcAddr, expAddr);
  SecondNormCompute<TOut>(dst, expBuf, reduceTmpBuf, tilingInfo.aSize, tilingInfo.rSize, tilingInfo.rAligned);
}

template <typename TIn>
__aicore__ inline void SoftmaxARSmallRComputeMax(__local_mem__ TIn *srcAddr,
                                                 AscendC::Reg::RegTensor<int32_t> &arangeReg,
                                                 AscendC::Reg::RegTensor<int32_t> &indexReg,
                                                 AscendC::Reg::RegTensor<float> &srcReg,
                                                 AscendC::Reg::RegTensor<float> &maxReg, AscendC::Reg::MaskReg &mask,
                                                 uint16_t rLoopTimes, uint32_t aOffset, uint32_t srcRAligned) {
  for (uint16_t rIdx = 0U; rIdx < rLoopTimes; ++rIdx) {
    AscendC::MicroAPI::Muls(indexReg, arangeReg, static_cast<int32_t>(srcRAligned), mask);
    AscendC::MicroAPI::Adds(indexReg, indexReg, static_cast<int32_t>(aOffset * srcRAligned + rIdx), mask);
    GatherAsFp32(srcAddr, srcReg, (AscendC::MicroAPI::RegTensor<uint32_t> &)indexReg, mask);
    AscendC::MicroAPI::Max(maxReg, maxReg, srcReg, mask);
  }
}

template <typename TIn>
__aicore__ inline void SoftmaxARSmallRComputeExpSumCache(
    __local_mem__ TIn *srcAddr, __local_mem__ float *expCacheAddr, AscendC::Reg::RegTensor<int32_t> &arangeReg,
    AscendC::Reg::RegTensor<int32_t> &indexReg, AscendC::Reg::RegTensor<float> &srcReg,
    AscendC::Reg::RegTensor<float> &maxReg, AscendC::Reg::RegTensor<float> &expReg,
    AscendC::Reg::RegTensor<float> &sumReg, AscendC::Reg::MaskReg &mask, uint16_t rLoopTimes, uint32_t aOffset,
    uint32_t srcRAligned) {
  AscendC::MicroAPI::Duplicate(sumReg, 0.0F, mask);
  for (uint16_t rIdx = 0U; rIdx < rLoopTimes; ++rIdx) {
    AscendC::MicroAPI::Muls(indexReg, arangeReg, static_cast<int32_t>(srcRAligned), mask);
    AscendC::MicroAPI::Adds(indexReg, indexReg, static_cast<int32_t>(aOffset * srcRAligned + rIdx), mask);
    GatherAsFp32(srcAddr, srcReg, (AscendC::MicroAPI::RegTensor<uint32_t> &)indexReg, mask);
    AscendC::MicroAPI::Sub(expReg, srcReg, maxReg, mask);
    AscendC::MicroAPI::Exp(expReg, expReg, mask);
    AscendC::MicroAPI::Add(sumReg, sumReg, expReg, mask);
    AscendC::MicroAPI::DataCopy(expCacheAddr + rIdx * VL_FP32, expReg, mask);
  }
}

template <typename TOut>
__aicore__ inline void SoftmaxARSmallRComputeNormalizeScatter(
    __local_mem__ TOut *dstAddr, __local_mem__ float *expCacheAddr, AscendC::Reg::RegTensor<int32_t> &arangeReg,
    AscendC::Reg::RegTensor<int32_t> &indexReg, AscendC::Reg::RegTensor<int16_t> &arangeRegI16,
    AscendC::Reg::RegTensor<int16_t> &outIndexRegI16, AscendC::Reg::RegTensor<float> &expReg,
    AscendC::Reg::RegTensor<float> &sumReg, AscendC::Reg::RegTensor<float> &divReg, AscendC::Reg::MaskReg &mask,
    uint16_t rLoopTimes, uint32_t aOffset, uint32_t outRAligned) {
  for (uint16_t rIdx = 0U; rIdx < rLoopTimes; ++rIdx) {
    AscendC::MicroAPI::DataCopy(expReg, expCacheAddr + rIdx * VL_FP32);
    AscendC::MicroAPI::Div(divReg, expReg, sumReg, mask);
    if constexpr (sizeof(TOut) == sizeof(uint32_t)) {
      AscendC::MicroAPI::Muls(indexReg, arangeReg, static_cast<int32_t>(outRAligned), mask);
      AscendC::MicroAPI::Adds(indexReg, indexReg, static_cast<int32_t>(aOffset * outRAligned + rIdx), mask);
      ScatterFromFp32(dstAddr, divReg, (AscendC::MicroAPI::RegTensor<uint32_t> &)indexReg, mask);
    } else {
      AscendC::MicroAPI::Muls(outIndexRegI16, arangeRegI16, static_cast<int16_t>(outRAligned), mask);
      AscendC::MicroAPI::Adds(outIndexRegI16, outIndexRegI16, static_cast<int16_t>(aOffset * outRAligned + rIdx), mask);
      ScatterFromFp32(dstAddr, divReg, (AscendC::MicroAPI::RegTensor<uint16_t> &)outIndexRegI16, mask);
    }
  }
}

template <typename TIn, typename TOut>
__aicore__ inline void SoftmaxARSmallRCompute(const AscendC::LocalTensor<TOut> &dst,
                                              const AscendC::LocalTensor<TIn> &src,
                                              const AscendC::LocalTensor<float> &expCache,
                                              const SoftmaxARTilingInfo &tilingInfo, uint32_t outRAligned) {
  const uint16_t aLoopTimes = static_cast<uint16_t>(CeilDiv(tilingInfo.aSize, VL_FP32));
  const uint16_t rLoopTimes = static_cast<uint16_t>(tilingInfo.rSize);
  const uint32_t srcRAligned = tilingInfo.rAligned;
  __local_mem__ TIn *srcAddr = (__local_mem__ TIn *)src.GetPhyAddr();
  __local_mem__ TOut *dstAddr = (__local_mem__ TOut *)dst.GetPhyAddr();
  __local_mem__ float *expCacheAddr = (__local_mem__ float *)expCache.GetPhyAddr();

  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<int32_t> arangeReg;
    AscendC::MicroAPI::RegTensor<int32_t> indexReg;
    AscendC::MicroAPI::RegTensor<int16_t> arangeRegI16;
    AscendC::MicroAPI::RegTensor<int16_t> outIndexRegI16;
    AscendC::MicroAPI::RegTensor<float> srcReg;
    AscendC::MicroAPI::RegTensor<float> maxReg;
    AscendC::MicroAPI::RegTensor<float> expReg;
    AscendC::MicroAPI::RegTensor<float> sumReg;
    AscendC::MicroAPI::RegTensor<float> divReg;
    AscendC::MicroAPI::MaskReg mask;
    AscendC::MicroAPI::Arange(arangeReg, 0);
    AscendC::MicroAPI::Arange(arangeRegI16, 0);

    for (uint16_t aLoopIdx = 0U; aLoopIdx < aLoopTimes; ++aLoopIdx) {
      const uint32_t aOffset = aLoopIdx * VL_FP32;
      uint32_t validA = tilingInfo.aSize - aOffset;
      validA = validA > VL_FP32 ? VL_FP32 : validA;
      mask = AscendC::MicroAPI::UpdateMask<float>(validA);
      AscendC::MicroAPI::Duplicate(maxReg, NEG_INFINITY, mask);
      SoftmaxARSmallRComputeMax(srcAddr, arangeReg, indexReg, srcReg, maxReg, mask, rLoopTimes, aOffset, srcRAligned);
      SoftmaxARSmallRComputeExpSumCache(srcAddr, expCacheAddr, arangeReg, indexReg, srcReg, maxReg, expReg, sumReg,
                                        mask, rLoopTimes, aOffset, srcRAligned);
      AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
      SoftmaxARSmallRComputeNormalizeScatter(dstAddr, expCacheAddr, arangeReg, indexReg, arangeRegI16, outIndexRegI16,
                                             expReg, sumReg, divReg, mask, rLoopTimes, aOffset, outRAligned);
    }
  }
}
}  // namespace SoftmaxRegBase

template <typename TIn, typename TOut>
__aicore__ inline void SoftmaxARFullLoadExtendImpl(const AscendC::LocalTensor<TOut> &dst,
                                                   const AscendC::LocalTensor<TIn> &src,
                                                   AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t (&shape)[2]) {
  static_assert(IsSameType<TIn, float>::value || IsSameType<TIn, half>::value || IsSameType<TIn, bfloat16_t>::value,
                "SoftmaxARFullLoadExtendImpl only supports float, half and bfloat16_t inputs.");
  static_assert(IsSameType<TOut, float>::value || IsSameType<TOut, half>::value || IsSameType<TOut, bfloat16_t>::value,
                "SoftmaxARFullLoadExtendImpl only supports float, half and bfloat16_t outputs.");
  const uint32_t a = shape[0];
  const uint32_t r = shape[1];
  if (a == 0U || r == 0U) {
    return;
  }

  SoftmaxRegBase::SoftmaxARTilingInfo tilingInfo = SoftmaxRegBase::GetSoftmaxARTilingInfo<TIn>(a, r);
  ASCENDC_ASSERT(tmp_buf.GetSize() >= tilingInfo.requiredTmpBytes, {
    KERNEL_LOG(KERNEL_ERROR, "softmax api tmp_buf is too small, tmp_buf is %u bytes, required is %u bytes!",
               tmp_buf.GetSize(), tilingInfo.requiredTmpBytes);
  });

  AscendC::LocalTensor<float> expBuf = tmp_buf.template ReinterpretCast<float>();
  expBuf.SetSize(tilingInfo.expBufElems);
  AscendC::LocalTensor<float> reduceTmpBuf =
      tmp_buf[tilingInfo.expBufElems * sizeof(float)].template ReinterpretCast<float>();
  reduceTmpBuf.SetSize(tilingInfo.reduceTmpElems);

  SoftmaxRegBase::SoftmaxARFullLoadCompute<TIn, TOut>(dst, src, expBuf, reduceTmpBuf, tilingInfo);
}

template <typename TIn, typename TOut>
__aicore__ inline void SoftmaxARSmallRExtendImpl(const AscendC::LocalTensor<TOut> &dst,
                                                 const AscendC::LocalTensor<TIn> &src,
                                                 AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t (&shape)[2]) {
  static_assert(IsSameType<TIn, float>::value || IsSameType<TIn, half>::value || IsSameType<TIn, bfloat16_t>::value,
                "SoftmaxARSmallRExtendImpl only supports float, half and bfloat16_t inputs.");
  static_assert(IsSameType<TOut, float>::value || IsSameType<TOut, half>::value || IsSameType<TOut, bfloat16_t>::value,
                "SoftmaxARSmallRExtendImpl only supports float, half and bfloat16_t outputs.");
  const uint32_t a = shape[0];
  const uint32_t r = shape[1];
  if (a == 0U || r == 0U) {
    return;
  }

  ASCENDC_ASSERT(r <= SoftmaxRegBase::AR_SMALL_R_THRESHOLD, {
    KERNEL_LOG(KERNEL_ERROR, "softmax small-r api only supports r <= %u, but r is %u!",
               SoftmaxRegBase::AR_SMALL_R_THRESHOLD, r);
  });

  SoftmaxRegBase::SoftmaxARTilingInfo tilingInfo = SoftmaxRegBase::GetSoftmaxARTilingInfo<TIn>(a, r);
  const uint32_t outRAligned = SoftmaxRegBase::CalcRAligned<TOut>(r);
  const uint32_t requiredTmpBytes = SoftmaxRegBase::CalcSmallRExpCacheBytes(r);
  ASCENDC_ASSERT(tmp_buf.GetSize() >= requiredTmpBytes, {
    KERNEL_LOG(KERNEL_ERROR, "softmax small-r api tmp_buf is too small, tmp_buf is %u bytes, required is %u bytes!",
               tmp_buf.GetSize(), requiredTmpBytes);
  });

  AscendC::LocalTensor<float> expCache = tmp_buf.template ReinterpretCast<float>();
  expCache.SetSize(SoftmaxRegBase::CalcSmallRExpCacheElems(r));

  SoftmaxRegBase::SoftmaxARSmallRCompute<TIn, TOut>(dst, src, expCache, tilingInfo, outRAligned);
}

template <typename TIn, typename TOut>
__aicore__ inline void SoftmaxARFullLoadExtend(const AscendC::LocalTensor<TOut> &dst,
                                               const AscendC::LocalTensor<TIn> &src,
                                               AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t (&shape)[2]) {
  const uint32_t a = shape[0];
  const uint32_t r = shape[1];
  switch (SoftmaxRegBase::GetSoftmaxARPathType(r)) {
    case SoftmaxRegBase::SoftmaxARPathType::SMALL_R:
      SoftmaxARSmallRExtendImpl<TIn, TOut>(dst, src, tmp_buf, shape);
      return;
    case SoftmaxRegBase::SoftmaxARPathType::MID_R:
      SoftmaxRegBase::SoftmaxARMidRCompute<TIn, TOut>(dst, src, tmp_buf.template ReinterpretCast<float>(),
                                                      SoftmaxRegBase::GetSoftmaxARTilingInfo<TIn>(a, r));
      return;
    case SoftmaxRegBase::SoftmaxARPathType::FULL_LOAD:
      SoftmaxARFullLoadExtendImpl<TIn, TOut>(dst, src, tmp_buf, shape);
      return;
    default:
      break;
  }
  SoftmaxARFullLoadExtendImpl<TIn, TOut>(dst, src, tmp_buf, shape);
}

#endif  // __ASCENDC_API_REGBASE_SOFTMAX_H__
