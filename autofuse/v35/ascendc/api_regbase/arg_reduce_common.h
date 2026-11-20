/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ARG_REDUCE_COMMON_H__
#define __ASCENDC_API_REGBASE_ARG_REDUCE_COMMON_H__

#include <cstdint>
#include <limits>
#include <type_traits>

namespace AscendC {
template <typename T, bool isMax>
struct ArgReducePolicy {
  static constexpr CMPMODE keepComparison = isMax ? CMPMODE::GE : CMPMODE::LE;
  static constexpr CMPMODE betterComparison = isMax ? CMPMODE::GT : CMPMODE::LT;
  static constexpr Reg::ReduceType valueReduction = isMax ? Reg::ReduceType::MAX : Reg::ReduceType::MIN;
  static constexpr T identity = std::is_same_v<T, float>
                                    ? (isMax ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity())
                                    : (isMax ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max());
};

// Shared register primitives for raw-input reduction and partial-result merging.
// 带外部 activeMask 的逐行合并（供 Phase2 向量化分块/尾块使用）。
// activeMask 由调用方按块内有效列数生成；value 与 index 位宽不同（如 float 4B / int64 8B）时，
// activeMask 只激活前 count 个 lane，所有 mask 运算均带 activeMask，多余 lane 不参与，
// 从而规避"mask 按 value 类型生成、作用于 index 寄存器错位"的问题（count ≤ 较小 vlSize）。
template <typename T, typename I, bool isMax>
__simd_callee__ inline void ArgReduceMergeMasked(Reg::RegTensor<T> &bestValue, Reg::RegTensor<I> &bestIndex,
                                                 Reg::RegTensor<T> &nextBestValue, Reg::RegTensor<I> &nextBestIndex,
                                                 Reg::MaskReg &activeMask) {
  Reg::MaskReg updateMask, valueMask, earlierIndex;
  Reg::Compare<I, CMPMODE::LT>(earlierIndex, nextBestIndex, bestIndex, activeMask);
  Reg::Compare<T, CMPMODE::EQ>(valueMask, nextBestValue, bestValue, activeMask);
  Reg::Compare<T, ArgReducePolicy<T, isMax>::betterComparison>(updateMask, nextBestValue, bestValue, activeMask);
  if constexpr (std::is_same_v<T, float>) {
    Reg::MaskReg bestNaN, nextNaN, nanMask;
    Reg::Compare<T, CMPMODE::NE>(bestNaN, bestValue, bestValue, activeMask);
    Reg::Compare<T, CMPMODE::NE>(nextNaN, nextBestValue, nextBestValue, activeMask);
    Reg::And(nanMask, bestNaN, nextNaN, activeMask);
    Reg::Or(valueMask, valueMask, nanMask, activeMask);
    Reg::Not(nanMask, bestNaN, activeMask);
    Reg::And(nanMask, nanMask, nextNaN, activeMask);
    Reg::Or(updateMask, updateMask, nanMask, activeMask);
  }
  Reg::And(valueMask, valueMask, earlierIndex, activeMask);
  Reg::Or(updateMask, updateMask, valueMask, activeMask);
  Reg::Select(bestValue, nextBestValue, bestValue, updateMask);
  Reg::Select(bestIndex, nextBestIndex, bestIndex, updateMask);
}

// Ordered R traversal keeps the earlier index when values compare equal.
template <typename T, typename I, bool isMax>
__simd_callee__ inline void ArgReduceUpdateInOrder(Reg::RegTensor<T> &bestValue, Reg::RegTensor<I> &bestIndex,
                                                   Reg::RegTensor<T> &currentValue, I current, Reg::MaskReg &mask) {
  Reg::MaskReg keepMask;
  Reg::Compare<T, ArgReducePolicy<T, isMax>::keepComparison>(keepMask, bestValue, currentValue, mask);
  if constexpr (std::is_same_v<T, float>) {
    Reg::MaskReg bestNaN;
    Reg::Compare<T, CMPMODE::NE>(bestNaN, bestValue, bestValue, mask);
    Reg::Or(keepMask, keepMask, bestNaN, mask);
  }
  Reg::RegTensor<I> currentIndex;
  Reg::Duplicate(currentIndex, current, mask);
  Reg::Select(bestValue, bestValue, currentValue, keepMask);
  Reg::Select(bestIndex, bestIndex, currentIndex, keepMask);
}

template <typename T, bool aligned = true>
__simd_callee__ inline void ArgReduceLoadRow(Reg::RegTensor<T> &dst, __ubuf__ T *src, uint32_t count) {
  constexpr uint32_t vlSize = GetVecLen() / sizeof(T);
  if (count == vlSize) {
    if constexpr (aligned) {
      Reg::LoadAlign(dst, src);
    } else {
      Reg::UnalignRegForLoad loadReg;
      Reg::LoadUnAlignPre(loadReg, src);
      Reg::LoadUnAlign(dst, loadReg, src);
    }
  } else {
    using OffsetT = std::conditional_t<sizeof(T) == 8, int64_t, int32_t>;
    using OffsetUintT = std::make_unsigned_t<OffsetT>;
    Reg::RegTensor<OffsetT> offset;
    Reg::MaskReg mask = Reg::UpdateMask<T>(count);
    Reg::Arange(offset, static_cast<OffsetT>(0));
    Reg::Gather(dst, src, (Reg::RegTensor<OffsetUintT> &)offset, mask);
  }
}

// 加载 int32 index 行切片到内部 index 寄存器（IndexRegT 与 value 位宽对齐）。
// value 8B / IndexRegT 8B：int32 → int64 零扩展（Interleave 与 0）；
// count 按源 int32 的 lane 数控制读取，混宽时也不能读取完整 int32 VL。
template <typename IndexRegT>
__simd_callee__ inline void ArgReduceLoadIndexBlock(Reg::RegTensor<IndexRegT> &dst, const __ubuf__ int32_t *indexSrc,
                                                    uint32_t rowBase, uint32_t count) {
  __ubuf__ int32_t *indexPtr = (__ubuf__ int32_t *)indexSrc;
  if constexpr (std::is_same_v<IndexRegT, int32_t>) {
    ArgReduceLoadRow(dst, indexPtr + rowBase, count);
  } else {
    Reg::RegTensor<int32_t> src32, zero32, lo, hi;
    ArgReduceLoadRow(src32, indexPtr + rowBase, count);
    Reg::Duplicate(zero32, 0);
    Reg::Interleave(lo, hi, src32, zero32);
    dst = (Reg::RegTensor<IndexRegT> &)lo;
  }
}

// 将内部 index 寄存器存储为 int32 的 count 个元素（IndexRegT 8B 时 DeInterleave 取低 32 位）。
template <typename IndexRegT>
__simd_callee__ inline void ArgReduceStoreIndexBlock(Reg::RegTensor<IndexRegT> &bestIndex, __ubuf__ int32_t *indexDst,
                                                     uint32_t a0, uint32_t count) {
  if constexpr (std::is_same_v<IndexRegT, int32_t>) {
    Reg::Store(indexDst + a0, bestIndex, count);
  } else {
    Reg::RegTensor<int32_t> lo, hi;
    Reg::DeInterleave(lo, hi, (Reg::RegTensor<int32_t> &)bestIndex, (Reg::RegTensor<int32_t> &)bestIndex);
    Reg::Store(indexDst + a0, lo, count);
  }
}

}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_ARG_REDUCE_COMMON_H__
