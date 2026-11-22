/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ARG_REDUCE_CORE_H__
#define __ASCENDC_API_REGBASE_ARG_REDUCE_CORE_H__

#include <cstdint>
#include <limits>
#include <type_traits>

#ifndef __ASCENDC_API_REGBASE_ARG_REDUCE_COMMON_H__
#include "arg_reduce_common.h"
#endif

namespace AscendC {
// Raw-input reduction algorithms: row-wise SIMD over R and column-wise SIMD over A.
// 将归约后的内部 index 寄存器（IndexRegT 与 value 位宽对齐）存储为 int32 的 1 个元素。
// 存储 dtype 统一为 int32；value 8B 时内部 index 寄存器为 int64（保证 mask lane 对齐），
// 此时用 DeInterleave 拆出低 32 位作为索引（索引 ≤ INT32_MAX）。
template <typename IndexRegT>
__simd_callee__ inline void ArgReduceStoreIndex(__ubuf__ int32_t *dst, Reg::RegTensor<IndexRegT> &idx) {
  if constexpr (std::is_same_v<IndexRegT, int32_t>) {
    Reg::Store(dst, idx, 1);
  } else {
    Reg::RegTensor<int32_t> lo, hi;
    Reg::DeInterleave(lo, hi, (Reg::RegTensor<int32_t> &)idx, (Reg::RegTensor<int32_t> &)idx);
    Reg::Store(dst, lo, 1);
  }
}

template <typename T>
__simd_callee__ inline void ArgReduceEq(Reg::MaskReg &eqMask, Reg::RegTensor<T> &curValue,
                                        Reg::RegTensor<T> &rowMaxBcast, Reg::MaskReg &rowNaN,
                                        Reg::MaskReg &activeMask) {
  Reg::Compare<T, CMPMODE::EQ>(eqMask, curValue, rowMaxBcast, activeMask);
  if constexpr (std::is_same_v<T, float>) {
    Reg::MaskReg nanMask;
    Reg::Compare<T, CMPMODE::NE>(nanMask, curValue, curValue, activeMask);
    Reg::And(nanMask, nanMask, rowNaN, activeMask);
    Reg::Or(eqMask, eqMask, nanMask, activeMask);
  }
}

template <typename T, bool isRa, bool isAligned>
__simd_callee__ inline void ArgReduceLoadBlock(Reg::RegTensor<T> &dst, __ubuf__ T *rowSrc, __ubuf__ T *src,
                                               uint32_t dimA, uint32_t row, uint32_t offset, bool fullBlock,
                                               Reg::MaskReg &activeMask) {
  using IndexT = std::conditional_t<(sizeof(T) == sizeof(int64_t)), int64_t, int32_t>;
  using IndexUintT = std::make_unsigned_t<IndexT>;
  if constexpr (isRa) {
    Reg::RegTensor<IndexT> gatherIndex;
    Reg::Arange(gatherIndex, static_cast<IndexT>(offset));
    Reg::Muls(gatherIndex, gatherIndex, static_cast<IndexT>(dimA), activeMask);
    Reg::Adds(gatherIndex, gatherIndex, static_cast<IndexT>(row), activeMask);
    Reg::Gather(dst, src, (Reg::RegTensor<IndexUintT> &)gatherIndex, activeMask);
  } else if (fullBlock) {
    if constexpr (isAligned) {
      Reg::LoadAlign(dst, rowSrc + offset);
    } else {
      Reg::UnalignRegForLoad loadReg;
      Reg::LoadUnAlignPre(loadReg, rowSrc + offset);
      Reg::LoadUnAlign(dst, loadReg, rowSrc + offset);
    }
  } else {
    Reg::RegTensor<IndexT> gatherIndex;
    Reg::Arange(gatherIndex, static_cast<IndexT>(offset));
    Reg::Gather(dst, rowSrc, (Reg::RegTensor<IndexUintT> &)gatherIndex, activeMask);
  }
}

template <typename T>
using ArgReduceIndexRegT = std::conditional_t<(sizeof(T) == sizeof(int64_t)), int64_t, int32_t>;

// 全块累加与尾块：沿 R 轴按自然顺序更新最佳值与块起点。仅做寄存器运算，不改变调用方状态。
template <typename T, bool isRa, bool isAligned, bool isMax>
__simd_callee__ inline void ArgReduceRowAccumulateBlocks(Reg::RegTensor<T> &bestValue,
                                                         Reg::RegTensor<ArgReduceIndexRegT<T>> &bestBlock,
                                                         Reg::RegTensor<T> &currentValue, __ubuf__ T *rowSrc,
                                                         __ubuf__ T *src, uint32_t dimA, uint32_t i, uint32_t indexBase,
                                                         uint32_t validR, Reg::RegTensor<T> &identity,
                                                         Reg::MaskReg &allMask) {
  using IndexRegT = ArgReduceIndexRegT<T>;
  constexpr uint32_t vlSize = GetVecLen() / sizeof(T);
  const uint32_t fullBlocks = validR / vlSize;
  // Bound the inner counter for hardware loops without truncating the public dimensions.
  for (uint32_t firstBlock = 1U; firstBlock < fullBlocks;) {
    const uint32_t remainingBlocks = fullBlocks - firstBlock;
    const uint16_t blockCount = static_cast<uint16_t>(remainingBlocks < UINT16_MAX ? remainingBlocks : UINT16_MAX);
    for (uint16_t block = 0U; block < blockCount; ++block) {
      const uint32_t offset = (firstBlock + static_cast<uint32_t>(block)) * vlSize;
      ArgReduceLoadBlock<T, isRa, isAligned>(currentValue, rowSrc, src, dimA, i, offset, true, allMask);
      const IndexRegT blockStart = static_cast<IndexRegT>(static_cast<uint64_t>(indexBase) + offset);
      ArgReduceUpdateInOrder<T, IndexRegT, isMax>(bestValue, bestBlock, currentValue, blockStart, allMask);
    }
    firstBlock += blockCount;
  }
  const uint32_t offset = fullBlocks * vlSize;
  if (fullBlocks != 0U && offset < validR) {
    uint32_t maskCount = validR - offset;
    Reg::MaskReg activeMask = Reg::UpdateMask<T>(maskCount);
    ArgReduceLoadBlock<T, isRa, isAligned>(currentValue, rowSrc, src, dimA, i, offset, false, activeMask);
    Reg::Select(currentValue, currentValue, identity, activeMask);
    const IndexRegT blockStart = static_cast<IndexRegT>(static_cast<uint64_t>(indexBase) + offset);
    ArgReduceUpdateInOrder<T, IndexRegT, isMax>(bestValue, bestBlock, currentValue, blockStart, allMask);
  }
}

// Value reduction follows the policy; NaN presence always uses a MAX over integer flags.
template <typename T, bool isMax>
__simd_callee__ inline void ArgReduceRowExtreme(Reg::RegTensor<T> &bestValue, Reg::RegTensor<T> &rowMaxReg,
                                                Reg::RegTensor<T> &finiteValue, Reg::RegTensor<T> &identity,
                                                Reg::RegTensor<T> &nanValue, Reg::RegTensor<int32_t> &one,
                                                Reg::RegTensor<int32_t> &zero, Reg::RegTensor<int32_t> &nanFlag,
                                                Reg::MaskReg &rowNaN, Reg::MaskReg &allMask) {
  if constexpr (std::is_same_v<T, float>) {
    Reg::Compare<T, CMPMODE::NE>(rowNaN, bestValue, bestValue, allMask);
    Reg::Select(finiteValue, identity, bestValue, rowNaN);
    Reg::Reduce<ArgReducePolicy<T, isMax>::valueReduction>(rowMaxReg, finiteValue, allMask);
    Reg::Select(nanFlag, one, zero, rowNaN);
    Reg::Reduce<Reg::ReduceType::MAX>(nanFlag, nanFlag, allMask);
    Reg::Compare<int32_t, CMPMODE::NE>(rowNaN, nanFlag, zero, allMask);
    Reg::Select(rowMaxReg, nanValue, rowMaxReg, rowNaN);
  } else {
    Reg::Reduce<ArgReducePolicy<T, isMax>::valueReduction>(rowMaxReg, bestValue, allMask);
  }
}

// 行内最小全局索引选择，并在 withValue 时按获胜索引回取原始 value。
template <typename T, bool isRa, bool withValue>
__simd_callee__ inline void ArgReduceRowSelectAndEmit(
    Reg::RegTensor<T> &bestValue, Reg::RegTensor<ArgReduceIndexRegT<T>> &bestBlock,
    Reg::RegTensor<ArgReduceIndexRegT<T>> &cand, Reg::RegTensor<ArgReduceIndexRegT<T>> &laneIndex,
    Reg::RegTensor<ArgReduceIndexRegT<T>> &bigReg, Reg::RegTensor<ArgReduceIndexRegT<T>> &rowIdxReg,
    Reg::RegTensor<ArgReduceIndexRegT<T>> &gatherIndex, Reg::RegTensor<T> &rowMaxReg, Reg::RegTensor<T> &rowMaxBcast,
    Reg::RegTensor<T> &currentValue, Reg::MaskReg &eqMask, Reg::MaskReg &rowNaN, Reg::MaskReg &activeMask,
    Reg::MaskReg &allMask, Reg::MaskReg &validLaneMask, __ubuf__ int32_t *indexDst, __ubuf__ T *valueDst,
    __ubuf__ T *rowSrc, uint32_t dimA, uint32_t i, uint32_t indexBase) {
  using IndexRegT = ArgReduceIndexRegT<T>;
  Reg::Duplicate(rowMaxBcast, rowMaxReg, allMask);
  if constexpr (std::is_same_v<T, float>) {
    Reg::Compare<T, CMPMODE::NE>(rowNaN, rowMaxBcast, rowMaxBcast, allMask);
  }
  ArgReduceEq<T>(eqMask, bestValue, rowMaxBcast, rowNaN, allMask);
  Reg::And(eqMask, eqMask, validLaneMask, allMask);
  Reg::Add(cand, bestBlock, laneIndex, allMask);
  Reg::Select(cand, cand, bigReg, eqMask);
  Reg::Reduce<Reg::ReduceType::MIN>(rowIdxReg, cand, allMask);
  if constexpr (withValue) {
    uint32_t valueCount = 1U;
    activeMask = Reg::UpdateMask<T>(valueCount);
    Reg::Adds(gatherIndex, rowIdxReg, -static_cast<IndexRegT>(indexBase), activeMask);
    if constexpr (isRa) {
      Reg::Muls(gatherIndex, gatherIndex, static_cast<IndexRegT>(dimA), activeMask);
      Reg::Adds(gatherIndex, gatherIndex, static_cast<IndexRegT>(i), activeMask);
    }
    Reg::Gather(currentValue, rowSrc, (Reg::RegTensor<std::make_unsigned_t<IndexRegT>> &)gatherIndex, activeMask);
    Reg::Store(valueDst + i, currentValue, 1);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
  }
  ArgReduceStoreIndex<IndexRegT>(indexDst + i, rowIdxReg);
}

// AR/RA 行归约：首块初始化后，后续块沿 R 轴按自然顺序更新最佳值和块起点。
// 行末再归约值和最小全局索引。尾块只通过 activeMask 参与加载和最终候选选择。
// dimA remains the physical RA row pitch; validA limits output columns without reading row padding.
template <typename T, bool isRa, bool isAligned, bool withValue, bool isMax>
__simd_vf__ inline void ArgReduceRowImpl(__ubuf__ int32_t *indexDst, __ubuf__ T *src, uint32_t dimA, uint32_t dimR,
                                         uint32_t indexBase, uint32_t validR, __ubuf__ T *valueDst = nullptr,
                                         uint32_t validA = 0U) {
  if (validR == 0U) {
    return;
  }
  constexpr uint32_t vlSize = GetVecLen() / sizeof(T);
  using IndexRegT = ArgReduceIndexRegT<T>;
  Reg::MaskReg allMask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
  Reg::RegTensor<T> currentValue, bestValue, rowMaxReg, rowMaxBcast, finiteValue, identity;
  Reg::MaskReg activeMask, eqMask, rowNaN;
  Reg::RegTensor<IndexRegT> rowIdxReg, bestBlock, cand, laneIndex, gatherIndex, bigReg;
  Reg::RegTensor<int32_t> one, zero, nanFlag;
  Reg::RegTensor<T> nanValue;
  Reg::Duplicate(identity, ArgReducePolicy<T, isMax>::identity, allMask);
  Reg::Duplicate(bigReg, std::numeric_limits<IndexRegT>::max(), allMask);
  Reg::Arange(laneIndex, 0);
  uint32_t validLaneCount = validR < vlSize ? validR : vlSize;
  Reg::MaskReg validLaneMask = Reg::UpdateMask<T>(validLaneCount);
  if constexpr (std::is_same_v<T, float>) {
    Reg::Duplicate(one, 1, allMask);
    Reg::Duplicate(zero, 0, allMask);
    Reg::Duplicate(nanValue, std::numeric_limits<T>::quiet_NaN(), allMask);
  }

  const uint32_t actualA = (validA == 0U || validA > dimA) ? dimA : validA;
  for (uint32_t i = 0; i < actualA; ++i) {
    const uint32_t rowBase = isRa ? 0U : i * dimR;
    __ubuf__ T *rowSrc = isRa ? src : src + rowBase;
    const bool firstFullBlock = validR >= vlSize;
    activeMask = firstFullBlock ? allMask : validLaneMask;
    ArgReduceLoadBlock<T, isRa, isAligned>(currentValue, rowSrc, src, dimA, i, 0U, firstFullBlock, activeMask);
    Reg::Select(bestValue, currentValue, identity, activeMask);
    Reg::Duplicate(bestBlock, static_cast<IndexRegT>(indexBase), allMask);
    ArgReduceRowAccumulateBlocks<T, isRa, isAligned, isMax>(bestValue, bestBlock, currentValue, rowSrc, src, dimA, i,
                                                            indexBase, validR, identity, allMask);
    ArgReduceRowExtreme<T, isMax>(bestValue, rowMaxReg, finiteValue, identity, nanValue, one, zero, nanFlag, rowNaN,
                                  allMask);
    ArgReduceRowSelectAndEmit<T, isRa, withValue>(
        bestValue, bestBlock, cand, laneIndex, bigReg, rowIdxReg, gatherIndex, rowMaxReg, rowMaxBcast, currentValue,
        eqMask, rowNaN, activeMask, allMask, validLaneMask, indexDst, valueDst, rowSrc, dimA, i, indexBase);
  }
}

// RA column reduction: each active lane owns one output column.
template <typename T, bool withValue, bool isMax>
__simd_vf__ inline void ArgReduceRaImpl(__ubuf__ int32_t *indexDst, __ubuf__ T *src, uint32_t dimA, uint32_t validR,
                                        uint32_t indexBase, __ubuf__ T *valueDst = nullptr) {
  using IndexT = std::conditional_t<sizeof(T) == sizeof(int64_t), int64_t, int32_t>;
  constexpr uint32_t vlSize = GetVecLen() / sizeof(T);
  Reg::RegTensor<T> bestValue, currentValue;
  Reg::RegTensor<IndexT> bestIndex;
  for (uint32_t a = 0; a < dimA; a += vlSize) {
    const uint32_t count = dimA - a < vlSize ? dimA - a : vlSize;
    uint32_t remaining = count;
    Reg::MaskReg mask = Reg::UpdateMask<T>(remaining);
    ArgReduceLoadRow(bestValue, src + a, count);
    Reg::Duplicate(bestIndex, static_cast<IndexT>(indexBase), mask);
    for (uint32_t r = 1; r < validR; ++r) {
      if (dimA % (ONE_BLK_SIZE / sizeof(T)) == 0) {
        ArgReduceLoadRow(currentValue, src + r * dimA + a, count);
      } else {
        ArgReduceLoadRow<T, false>(currentValue, src + r * dimA + a, count);
      }
      ArgReduceUpdateInOrder<T, IndexT, isMax>(bestValue, bestIndex, currentValue, static_cast<IndexT>(indexBase + r),
                                               mask);
    }
    if constexpr (withValue) {
      Reg::Store(valueDst + a, bestValue, count);
      Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
    }
    ArgReduceStoreIndexBlock(bestIndex, indexDst, a, count);
  }
}

// Full-VL RA keeps optional historical accumulation inside the same VF.
template <typename T, bool withValue, uint32_t aGroups, bool accumulate, bool isMax>
__simd_vf__ inline void ArgReduceRaFullImpl(__ubuf__ int32_t *indexDst, __ubuf__ T *src, uint32_t dimA, uint32_t validR,
                                            uint32_t indexBase, __ubuf__ T *valueDst = nullptr, bool firstTile = true) {
  using IndexT = std::conditional_t<sizeof(T) == sizeof(int64_t), int64_t, int32_t>;
  constexpr uint32_t vlSize = GetVecLen() / sizeof(T);
  Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
  Reg::RegTensor<T> bestValue[aGroups], currentValue[aGroups];
  Reg::RegTensor<IndexT> bestIndex[aGroups];
  // A and R tiles fit in UB. 16-bit VF counters enable hardware loop generation.
  const uint16_t groups = static_cast<uint16_t>(dimA / (aGroups * vlSize));
  for (uint16_t block = 0; block < groups; ++block) {
    const uint32_t a = static_cast<uint32_t>(block) * aGroups * vlSize;
#pragma unroll
    for (uint32_t g = 0; g < aGroups; ++g) {
      Reg::LoadAlign(bestValue[g], src + a + g * vlSize);
      Reg::Duplicate(bestIndex[g], static_cast<IndexT>(0), mask);
    }
    for (uint16_t r = 1; r < static_cast<uint16_t>(validR); ++r) {
#pragma unroll
      for (uint32_t g = 0; g < aGroups; ++g) {
        Reg::LoadAlign(currentValue[g], src + r * dimA + a + g * vlSize);
      }
#pragma unroll
      for (uint32_t g = 0; g < aGroups; ++g) {
        ArgReduceUpdateInOrder<T, IndexT, isMax>(bestValue[g], bestIndex[g], currentValue[g], static_cast<IndexT>(r),
                                                 mask);
      }
    }
#pragma unroll
    for (uint32_t g = 0; g < aGroups; ++g) {
      Reg::Adds(bestIndex[g], bestIndex[g], static_cast<IndexT>(indexBase), mask);
      if constexpr (accumulate) {
        if (!firstTile) {
          Reg::RegTensor<T> historyValue;
          Reg::RegTensor<IndexT> historyIndex;
          Reg::LoadAlign(historyValue, valueDst + a + g * vlSize);
          ArgReduceLoadIndexBlock(historyIndex, indexDst, a + g * vlSize, vlSize);
          ArgReduceMergeMasked<T, IndexT, isMax>(bestValue[g], bestIndex[g], historyValue, historyIndex, mask);
        }
      }
      if constexpr (withValue) {
        Reg::Store(valueDst + a + g * vlSize, bestValue[g], vlSize);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
      }
      ArgReduceStoreIndexBlock(bestIndex[g], indexDst, a + g * vlSize, vlSize);
    }
  }
}

}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_ARG_REDUCE_CORE_H__
