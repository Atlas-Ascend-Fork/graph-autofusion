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

#include <cstdint>
#include <limits>
#include <type_traits>

namespace AscendC {
template <typename T>
__simd_vf__ inline void ArgmaxCopyImpl(__ubuf__ T *dst, __ubuf__ T *src, uint32_t count) {
  constexpr uint32_t vlSize = GetVecLen() / sizeof(T);
  using OffsetT = std::conditional_t<sizeof(T) == 8, int64_t, int32_t>;
  using OffsetUintT = std::make_unsigned_t<OffsetT>;
  Reg::RegTensor<T> value;
  Reg::RegTensor<OffsetT> offset;
  for (uint32_t start = 0; start < count; start += vlSize) {
    uint32_t remaining = count - start;
    Reg::MaskReg mask = Reg::UpdateMask<T>(remaining);
    Reg::Arange(offset, static_cast<OffsetT>(start));
    Reg::Gather(value, src, (Reg::RegTensor<OffsetUintT> &)offset, mask);
    Reg::Store(dst + start, value, count - start < vlSize ? count - start : vlSize);
  }
}

template <typename T>
__aicore__ inline void ArgmaxCopy(const LocalTensor<T> &dst, const LocalTensor<T> &src, uint32_t count) {
  asc_vf_call<ArgmaxCopyImpl<T>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), count);
}

// 将归约后的内部 index 寄存器（IndexRegT 与 value 位宽对齐）存储为 int32 的 1 个元素。
// 存储 dtype 统一为 int32；value 8B 时内部 index 寄存器为 int64（保证 mask lane 对齐），
// 此时用 DeInterleave 拆出低 32 位作为索引（索引 ≤ INT32_MAX）。
template <typename IndexRegT>
__simd_callee__ inline void ArgmaxStoreIndex(__ubuf__ int32_t *dst, Reg::RegTensor<IndexRegT> &idx) {
  if constexpr (std::is_same_v<IndexRegT, int32_t>) {
    Reg::Store(dst, idx, 1);
  } else {
    Reg::RegTensor<int32_t> lo, hi;
    Reg::DeInterleave(lo, hi, (Reg::RegTensor<int32_t> &)idx, (Reg::RegTensor<int32_t> &)idx);
    Reg::Store(dst, lo, 1);
  }
}

template <typename T>
__simd_callee__ inline void ArgmaxFoldMax(Reg::RegTensor<T> &bestValue, Reg::RegTensor<T> &currentValue,
                                          Reg::RegTensor<T> &lowReg, Reg::MaskReg &seenNaN, Reg::MaskReg &activeMask,
                                          Reg::MaskReg &allMask) {
  if constexpr (std::is_same_v<T, float>) {
    Reg::MaskReg nanMask;
    Reg::Compare<T, CMPMODE::NE>(nanMask, currentValue, currentValue, activeMask);
    Reg::Or(seenNaN, seenNaN, nanMask, allMask);
    Reg::Select(currentValue, lowReg, currentValue, nanMask);
  }
  Reg::Max<T, Reg::MaskMergeMode::ZEROING>(bestValue, bestValue, currentValue, allMask);
}

template <typename T>
__simd_callee__ inline void ArgmaxEq(Reg::MaskReg &eqMask, Reg::RegTensor<T> &curValue, Reg::RegTensor<T> &rowMaxBcast,
                                     Reg::MaskReg &rowNaN, Reg::MaskReg &activeMask) {
  Reg::Compare<T, CMPMODE::EQ>(eqMask, curValue, rowMaxBcast, activeMask);
  if constexpr (std::is_same_v<T, float>) {
    Reg::MaskReg nanMask;
    Reg::Compare<T, CMPMODE::NE>(nanMask, curValue, curValue, activeMask);
    Reg::And(nanMask, nanMask, rowNaN, activeMask);
    Reg::Or(eqMask, eqMask, nanMask, activeMask);
  }
}

// 带外部 activeMask 的逐行合并（供 Phase2 向量化分块/尾块使用）：语义与 ArgmaxMerge 完全一致。
// activeMask 由调用方按块内有效列数生成；value 与 index 位宽不同（如 float 4B / int64 8B）时，
// activeMask 只激活前 count 个 lane，所有 mask 运算均带 activeMask，多余 lane 不参与，
// 从而规避"mask 按 value 类型生成、作用于 index 寄存器错位"的问题（count ≤ 较小 vlSize）。
template <typename T, typename I>
__simd_callee__ inline void ArgmaxMergeMasked(Reg::RegTensor<T> &bestValue, Reg::RegTensor<I> &bestIndex,
                                              Reg::RegTensor<T> &nextBestValue, Reg::RegTensor<I> &nextBestIndex,
                                              Reg::MaskReg &activeMask) {
  Reg::MaskReg updateMask, valueMask, earlierIndex;
  Reg::Compare<I, CMPMODE::LT>(earlierIndex, nextBestIndex, bestIndex, activeMask);
  Reg::Compare<T, CMPMODE::EQ>(valueMask, nextBestValue, bestValue, activeMask);
  Reg::Compare<T, CMPMODE::GT>(updateMask, nextBestValue, bestValue, activeMask);
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

// AR/RA 行归约（两遍法，全向量、零标量）：
//   pass1：逐 block 用元素级 Max 折叠，行末 Reduce<MAX> 得行最大值；
//   pass2：广播行最大值，逐 block Compare(EQ) 后 Select 出全局索引候选，再 Reduce<MIN>
//          取最小全局索引（值等取最小 index 的确定性红线由 Min 保证）。
// 相比原“逐 block Arange + ArgmaxUpdate 维护 lane 级 index”，省掉每 block 的 Arange/Compare/Select
// 索引跟踪；代价是源 UB 读两遍（与既有双 vf 归约同量级）。
template <typename T, bool isRa, bool isAligned = false, bool withValue = false, bool withIndex = true>
__simd_vf__ inline void ArgmaxImpl(__ubuf__ int32_t *indexDst, __ubuf__ T *src, uint32_t dimA, uint32_t dimR,
                                   uint32_t mainCount, uint32_t indexBase, uint32_t validR,
                                   __ubuf__ T *valueDst = nullptr) {
  constexpr uint32_t vlSize = GetVecLen() / sizeof(T);
  // 内部 index 寄存器与 value 位宽对齐（保证 mask lane 数一致）；存储 dtype 仍为 int32。
  using IndexRegT = std::conditional_t<(sizeof(T) == 8), int64_t, int32_t>;
  using IndexUintT = std::conditional_t<(sizeof(T) == 8), uint64_t, uint32_t>;
  const T lowest = std::is_same_v<T, float> ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
  Reg::MaskReg allMask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
  IndexRegT bigIndex = std::numeric_limits<IndexRegT>::max();

  Reg::RegTensor<T> currentValue, bestValue, rowMaxReg, rowMaxBcast, lowReg;
  Reg::MaskReg activeMask, eqMask, rowNaN;
  Reg::RegTensor<IndexRegT> rowIdxReg, minCand, cand, globalIdxVec, gatherIndex;
  Reg::RegTensor<IndexRegT> bigReg;
  Reg::RegTensor<int32_t> one, zero, nanFlag;
  Reg::RegTensor<T> nanValue;
  Reg::Duplicate(lowReg, lowest, allMask);
  if constexpr (withIndex) {
    Reg::Duplicate(bigReg, bigIndex, allMask);
  }
  if constexpr (std::is_same_v<T, float>) {
    Reg::Duplicate(one, 1, allMask);
    Reg::Duplicate(zero, 0, allMask);
    Reg::Duplicate(nanValue, std::numeric_limits<T>::quiet_NaN(), allMask);
  }
  uint32_t mainEnd = 0;
  if constexpr (!isRa) {
    const uint32_t validMainCount = validR / vlSize;
    mainEnd = (mainCount < validMainCount ? mainCount : validMainCount) * vlSize;
  }

  for (uint32_t i = 0; i < dimA; ++i) {
    uint32_t rowBase = 0;
    __ubuf__ T *rowSrc = src;
    if constexpr (!isRa) {
      rowBase = i * dimR;
      rowSrc += rowBase;
    }
    // ---------------- pass1: 行最大值 ----------------
    Reg::Duplicate(bestValue, lowest, allMask);
    if constexpr (std::is_same_v<T, float>) {
      Reg::Compare<T, CMPMODE::NE>(rowNaN, lowReg, lowReg, allMask);
    }
    uint32_t offset = 0;
    if constexpr (!isRa && isAligned) {
      // 对齐主循环：连续整块 LoadAlign（行内不越界；只处理完整落在 validR 内的块）
      for (; offset < mainEnd; offset += vlSize) {
        Reg::LoadAlign(currentValue, rowSrc + offset);
        ArgmaxFoldMax(bestValue, currentValue, lowReg, rowNaN, allMask, allMask);
      }
    } else if constexpr (!isRa) {
      // 非对齐主循环：连续整块 LoadUnAlign（整 tile 在行内，不越界）
      Reg::UnalignRegForLoad loadReg;
      for (; offset < mainEnd; offset += vlSize) {
        Reg::LoadUnAlignPre(loadReg, rowSrc + offset);
        Reg::LoadUnAlign(currentValue, loadReg, rowSrc + offset);
        ArgmaxFoldMax(bestValue, currentValue, lowReg, rowNaN, allMask, allMask);
      }
    }
    // 尾块（RA 为整行；AR 为尾部）：Gather 逐块取数，mask 关闭越界 lane。
    uint32_t remaining = validR - mainEnd;
    for (; remaining > 0; offset += vlSize) {
      activeMask = Reg::UpdateMask<T>(remaining);
      if constexpr (isRa) {
        Reg::Arange(gatherIndex, static_cast<IndexRegT>(offset));
        Reg::Muls(gatherIndex, gatherIndex, static_cast<IndexRegT>(dimA), activeMask);
        Reg::Adds(gatherIndex, gatherIndex, static_cast<IndexRegT>(i), activeMask);
        Reg::Gather(currentValue, src, (Reg::RegTensor<IndexUintT> &)gatherIndex, activeMask);
      } else {
        Reg::Arange(gatherIndex, static_cast<IndexRegT>(rowBase + offset));
        Reg::Gather(currentValue, src, (Reg::RegTensor<IndexUintT> &)gatherIndex, activeMask);
      }
      Reg::Select(currentValue, currentValue, lowReg, activeMask);
      ArgmaxFoldMax(bestValue, currentValue, lowReg, rowNaN, activeMask, allMask);
    }
    Reg::Reduce<Reg::ReduceType::MAX>(rowMaxReg, bestValue, allMask);
    if constexpr (std::is_same_v<T, float>) {
      Reg::Select(nanFlag, one, zero, rowNaN);
      Reg::Reduce<Reg::ReduceType::MAX>(nanFlag, nanFlag, allMask);
      Reg::Compare<int32_t, CMPMODE::NE>(rowNaN, nanFlag, zero, allMask);
      Reg::Select(rowMaxReg, nanValue, rowMaxReg, rowNaN);
    }
    if constexpr (withValue && !withIndex) {
      Reg::Store(valueDst + i, rowMaxReg, 1);
    }
    if constexpr (!withIndex) {
      // 仅输出 value 的 vf：跳过 pass2，保证每个 vf 只 Store 一次。
      continue;
    }
    Reg::Duplicate(rowMaxBcast, rowMaxReg, allMask);
    if constexpr (std::is_same_v<T, float>) {
      Reg::Compare<T, CMPMODE::NE>(rowNaN, rowMaxBcast, rowMaxBcast, allMask);
    }

    // ---------------- pass2: 最小全局索引 ----------------
    Reg::Duplicate(minCand, bigIndex, allMask);
    offset = 0;
    if constexpr (!isRa && isAligned) {
      for (; offset < mainEnd; offset += vlSize) {
        Reg::LoadAlign(currentValue, rowSrc + offset);
        ArgmaxEq<T>(eqMask, currentValue, rowMaxBcast, rowNaN, allMask);
        Reg::Arange(globalIdxVec, static_cast<IndexRegT>(indexBase + offset));
        Reg::Select(cand, globalIdxVec, bigReg, eqMask);
        Reg::Min(minCand, minCand, cand, allMask);
      }
    } else if constexpr (!isRa) {
      Reg::UnalignRegForLoad loadReg;
      for (; offset < mainEnd; offset += vlSize) {
        Reg::LoadUnAlignPre(loadReg, rowSrc + offset);
        Reg::LoadUnAlign(currentValue, loadReg, rowSrc + offset);
        ArgmaxEq<T>(eqMask, currentValue, rowMaxBcast, rowNaN, allMask);
        Reg::Arange(globalIdxVec, static_cast<IndexRegT>(indexBase + offset));
        Reg::Select(cand, globalIdxVec, bigReg, eqMask);
        Reg::Min(minCand, minCand, cand, allMask);
      }
    }
    remaining = validR - mainEnd;
    for (; remaining > 0; offset += vlSize) {
      activeMask = Reg::UpdateMask<T>(remaining);
      if constexpr (isRa) {
        Reg::Arange(gatherIndex, static_cast<IndexRegT>(offset));
        Reg::Muls(gatherIndex, gatherIndex, static_cast<IndexRegT>(dimA), activeMask);
        Reg::Adds(gatherIndex, gatherIndex, static_cast<IndexRegT>(i), activeMask);
        Reg::Gather(currentValue, src, (Reg::RegTensor<IndexUintT> &)gatherIndex, activeMask);
      } else {
        Reg::Arange(gatherIndex, static_cast<IndexRegT>(rowBase + offset));
        Reg::Gather(currentValue, src, (Reg::RegTensor<IndexUintT> &)gatherIndex, activeMask);
      }
      ArgmaxEq<T>(eqMask, currentValue, rowMaxBcast, rowNaN, activeMask);
      Reg::Arange(globalIdxVec, static_cast<IndexRegT>(indexBase + offset));
      Reg::Select(cand, globalIdxVec, bigReg, eqMask);
      Reg::Min(minCand, minCand, cand, allMask);
    }
    Reg::Reduce<Reg::ReduceType::MIN>(rowIdxReg, minCand, allMask);
    if constexpr (withValue) {
      // 按获胜索引回取原始 value，保留首个最大元素的零符号位及 NaN payload。
      // rowIdxReg 包含全局索引，访问当前 tile 前先扣除 indexBase。
      uint32_t valueCount = 1U;
      activeMask = Reg::UpdateMask<T>(valueCount);
      Reg::Adds(gatherIndex, rowIdxReg, -static_cast<IndexRegT>(indexBase), activeMask);
      if constexpr (isRa) {
        Reg::Muls(gatherIndex, gatherIndex, static_cast<IndexRegT>(dimA), activeMask);
        Reg::Adds(gatherIndex, gatherIndex, static_cast<IndexRegT>(i), activeMask);
      }
      Reg::Gather(currentValue, rowSrc, (Reg::RegTensor<IndexUintT> &)gatherIndex, activeMask);
      Reg::Store(valueDst + i, currentValue, 1);
      Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
    }
    ArgmaxStoreIndex<IndexRegT>(indexDst + i, rowIdxReg);
  }
}

template <typename T, bool aligned = true>
__simd_callee__ inline void ArgmaxLoadRow(Reg::RegTensor<T> &dst, __ubuf__ T *src, uint32_t count) {
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
__simd_callee__ inline void ArgmaxPhase2LoadIndexRow(Reg::RegTensor<IndexRegT> &dst, const __ubuf__ int32_t *indexSrc,
                                                     uint32_t rowBase, uint32_t count) {
  __ubuf__ int32_t *indexPtr = (__ubuf__ int32_t *)indexSrc;
  if constexpr (std::is_same_v<IndexRegT, int32_t>) {
    ArgmaxLoadRow(dst, indexPtr + rowBase, count);
  } else {
    Reg::RegTensor<int32_t> src32, zero32, lo, hi;
    ArgmaxLoadRow(src32, indexPtr + rowBase, count);
    Reg::Duplicate(zero32, 0);
    Reg::Interleave(lo, hi, src32, zero32);
    dst = (Reg::RegTensor<IndexRegT> &)lo;
  }
}

// 将内部 index 寄存器存储为 int32 的 count 个元素（IndexRegT 8B 时 DeInterleave 取低 32 位）。
template <typename IndexRegT>
__simd_callee__ inline void ArgmaxPhase2StoreIndexRow(Reg::RegTensor<IndexRegT> &bestIndex, __ubuf__ int32_t *indexDst,
                                                      uint32_t a0, uint32_t count) {
  if constexpr (std::is_same_v<IndexRegT, int32_t>) {
    Reg::Store(indexDst + a0, bestIndex, count);
  } else {
    Reg::RegTensor<int32_t> lo, hi;
    Reg::DeInterleave(lo, hi, (Reg::RegTensor<int32_t> &)bestIndex, (Reg::RegTensor<int32_t> &)bestIndex);
    Reg::Store(indexDst + a0, lo, count);
  }
}

template <typename T, bool withValue>
__simd_vf__ inline void ArgmaxRaVectorImpl(__ubuf__ int32_t *indexDst, __ubuf__ T *src, uint32_t dimA, uint32_t validR,
                                           uint32_t indexBase, __ubuf__ T *valueDst = nullptr) {
  using IndexT = std::conditional_t<sizeof(T) == 8, int64_t, int32_t>;
  constexpr uint32_t vlSize = GetVecLen() / sizeof(T);
  Reg::RegTensor<T> bestValue, currentValue;
  Reg::RegTensor<IndexT> bestIndex, currentIndex;
  for (uint32_t a = 0; a < dimA; a += vlSize) {
    const uint32_t count = dimA - a < vlSize ? dimA - a : vlSize;
    uint32_t remaining = count;
    Reg::MaskReg mask = Reg::UpdateMask<T>(remaining);
    ArgmaxLoadRow(bestValue, src + a, count);
    Reg::Duplicate(bestIndex, static_cast<IndexT>(indexBase), mask);
    for (uint32_t r = 1; r < validR; ++r) {
      if (dimA % (32U / sizeof(T)) == 0) {
        ArgmaxLoadRow(currentValue, src + r * dimA + a, count);
      } else {
        ArgmaxLoadRow<T, false>(currentValue, src + r * dimA + a, count);
      }
      Reg::MaskReg updateMask;
      Reg::Compare<T, CMPMODE::GT>(updateMask, currentValue, bestValue, mask);
      if constexpr (std::is_same_v<T, float>) {
        Reg::MaskReg bestNaN, currentNaN;
        Reg::Compare<T, CMPMODE::NE>(bestNaN, bestValue, bestValue, mask);
        Reg::Compare<T, CMPMODE::NE>(currentNaN, currentValue, currentValue, mask);
        Reg::Not(bestNaN, bestNaN, mask);
        Reg::And(currentNaN, currentNaN, bestNaN, mask);
        Reg::Or(updateMask, updateMask, currentNaN, mask);
      }
      Reg::Select(bestValue, currentValue, bestValue, updateMask);
      Reg::Duplicate(currentIndex, static_cast<IndexT>(indexBase + r), mask);
      Reg::Select(bestIndex, currentIndex, bestIndex, updateMask);
    }
    if constexpr (withValue) {
      Reg::Store(valueDst + a, bestValue, count);
      Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
    }
    ArgmaxPhase2StoreIndexRow(bestIndex, indexDst, a, count);
  }
}

// ArgmaxMergeRound 向量化 vf：value/index 同一轮完成选择并写回。
// value 与 index 的 Store 之间插入 LocalMemBar，保证不同类型输出的写回顺序。
// 块内列数取 min(vlSize(value), 64)，activeMask 只激活 count 个 lane。
template <typename U>
__simd_vf__ inline void ArgmaxMergeRoundVectorIndexImpl(__ubuf__ U *bestValue, __ubuf__ int32_t *bestIndex,
                                                        __ubuf__ U *curValue, __ubuf__ int32_t *curIndex,
                                                        uint32_t dimA) {
  using IndexRegT = std::conditional_t<(sizeof(U) == 8), int64_t, int32_t>;
  constexpr uint32_t valueVl = GetVecLen() / sizeof(U);
  constexpr uint32_t indexVl = GetVecLen() / sizeof(IndexRegT);
  constexpr uint32_t step = valueVl < indexVl ? valueVl : indexVl;
  Reg::RegTensor<U> bestV, curV;
  Reg::RegTensor<IndexRegT> bestI, curI;
  if constexpr (sizeof(U) == 8) {
    Reg::RegTensor<int32_t> bestI32, curI32, src32, zero32, lo, hi;
    Reg::MaskReg indexAllMask = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zero32, 0, indexAllMask);
    for (uint32_t a0 = 0; a0 < dimA; a0 += step) {
      uint32_t count = (dimA - a0 < step) ? (dimA - a0) : step;
      uint32_t maskCount = count;
      Reg::MaskReg mask = Reg::UpdateMask<U>(maskCount);
      ArgmaxLoadRow(bestV, bestValue + a0, count);
      ArgmaxLoadRow(curV, curValue + a0, count);
      ArgmaxLoadRow(bestI32, bestIndex + a0, count);
      ArgmaxLoadRow(curI32, curIndex + a0, count);
      Reg::Interleave(lo, hi, bestI32, zero32);
      bestI = (Reg::RegTensor<IndexRegT> &)lo;
      Reg::Interleave(lo, hi, curI32, zero32);
      curI = (Reg::RegTensor<IndexRegT> &)lo;
      ArgmaxMergeMasked<U, IndexRegT>(bestV, bestI, curV, curI, mask);
      Reg::Store(bestValue + a0, bestV, count);
      Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
      Reg::DeInterleave(lo, hi, (Reg::RegTensor<int32_t> &)bestI, (Reg::RegTensor<int32_t> &)bestI);
      Reg::Store(bestIndex + a0, lo, count);
    }
    return;
  }
  for (uint32_t a0 = 0; a0 < dimA; a0 += step) {
    uint32_t count = (dimA - a0 < step) ? (dimA - a0) : step;
    uint32_t maskCount = count;
    Reg::MaskReg mask = Reg::UpdateMask<U>(maskCount);
    ArgmaxLoadRow(bestV, bestValue + a0, count);
    ArgmaxLoadRow(curV, curValue + a0, count);
    ArgmaxPhase2LoadIndexRow<IndexRegT>(bestI, bestIndex, a0, count);
    ArgmaxPhase2LoadIndexRow<IndexRegT>(curI, curIndex, a0, count);
    ArgmaxMergeMasked<U, IndexRegT>(bestV, bestI, curV, curI, mask);
    Reg::Store(bestValue + a0, bestV, count);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
    ArgmaxPhase2StoreIndexRow<IndexRegT>(bestI, bestIndex, a0, count);
  }
}

// 增量合并一次 R 轴分块归约结果（供 kCommon 模板 R 轴分块 multi_reduce 使用）：
// 与 Phase2/ArgmaxMerge 的确定性语义对齐：值大者胜；值等且 index 小者胜。
// 同宽/混宽、主块/尾块统一走向量化合并（无 GetValue/SetValue 标量路径）。
template <typename U>
__aicore__ inline void ArgmaxMergeRound(const LocalTensor<U> &bestValue, const LocalTensor<int32_t> &bestIndex,
                                        const LocalTensor<U> &curValue, const LocalTensor<int32_t> &curIndex,
                                        uint32_t dimA) {
  asc_vf_call<ArgmaxMergeRoundVectorIndexImpl<U>>(
      (__ubuf__ U *)bestValue.GetPhyAddr(), (__ubuf__ int32_t *)bestIndex.GetPhyAddr(),
      (__ubuf__ U *)curValue.GetPhyAddr(), (__ubuf__ int32_t *)curIndex.GetPhyAddr(), dimA);
}

template <typename U, class pattern>
__aicore__ inline void ArgMaxExtendImpl(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &src,
                                        const uint32_t *srcShape, uint32_t validR = 0U) {
  const uint32_t dimA = SupportType<pattern, Pattern::Reduce::AR>() ? srcShape[0] : srcShape[1];
  const uint32_t dimR = SupportType<pattern, Pattern::Reduce::AR>() ? srcShape[1] : srcShape[0];
  constexpr uint32_t vlSize = GetVecLen() / sizeof(U);
  const uint32_t actualValidR = (validR == 0U || validR > dimR) ? dimR : validR;
  __ubuf__ int32_t *localIndexDst = (__ubuf__ int32_t *)indexDst.GetPhyAddr();
  if constexpr (SupportType<pattern, Pattern::Reduce::AR>()) {
    for (; dimR % (32U / sizeof(U)) == 0;) {
      uint32_t mainCount = dimR / (2 * vlSize) * 2;
      asc_vf_call<ArgmaxImpl<U, false, true>>(localIndexDst, (__ubuf__ U *)src.GetPhyAddr(), dimA, dimR, mainCount, 0U,
                                              actualValidR, nullptr);
      break;
    }
    for (; dimR % (32U / sizeof(U)) != 0;) {
      uint32_t mainCount = dimR / vlSize;
      asc_vf_call<ArgmaxImpl<U, false>>(localIndexDst, (__ubuf__ U *)src.GetPhyAddr(), dimA, dimR, mainCount, 0U,
                                        actualValidR, nullptr);
      break;
    }
  } else {
    if (dimA >= vlSize || (actualValidR <= vlSize && actualValidR <= 2U * dimA)) {
      asc_vf_call<ArgmaxRaVectorImpl<U, false>>(localIndexDst, (__ubuf__ U *)src.GetPhyAddr(), dimA, actualValidR, 0U,
                                                nullptr);
      return;
    }
    asc_vf_call<ArgmaxImpl<U, true>>(localIndexDst, (__ubuf__ U *)src.GetPhyAddr(), dimA, dimR, 0, 0U, actualValidR,
                                     nullptr);
  }
}

/**
 * 输出局部最大值（valueDst）与其索引（indexDst）。两处使用：
 *   - 前端 ArgmaxWithValue 单算子：一次归约整个 R 轴，indexBase=0；
 *   - R 轴分核 Phase1：每核沿自己那一段 R_local 归约，indexBase=核偏移，使输出的 index 直接全局化。
 *
 * 实现说明：单个 asc_vf_call 内完成 value/index 输出：pass1 求最大值、pass2 选取首个索引，
 * 再按索引回取原始 value，依次 Store value/index，避免丢失零符号位和 NaN payload。
 */
template <typename U, class pattern>
__aicore__ inline void ArgMaxWithValueExtendImpl(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &valueDst,
                                                 const LocalTensor<U> &src, const LocalTensor<uint8_t> &sharedTmpBuffer,
                                                 const uint32_t *srcShape, uint32_t indexBase = 0U,
                                                 uint32_t validR = 0U) {
  const uint32_t dimA = SupportType<pattern, Pattern::Reduce::AR>() ? srcShape[0] : srcShape[1];
  const uint32_t dimR = SupportType<pattern, Pattern::Reduce::AR>() ? srcShape[1] : srcShape[0];
  constexpr uint32_t vlSize = GetVecLen() / sizeof(U);
  const uint32_t actualValidR = (validR == 0U || validR > dimR) ? dimR : validR;
  __ubuf__ int32_t *localIndexDst = (__ubuf__ int32_t *)indexDst.GetPhyAddr();
  __ubuf__ U *localValueDst = (__ubuf__ U *)valueDst.GetPhyAddr();
  constexpr bool isAr = SupportType<pattern, Pattern::Reduce::AR>();
  if constexpr (isAr) {
    for (; dimR % (32U / sizeof(U)) == 0;) {
      uint32_t mainCount = dimR / (2 * vlSize) * 2;
      asc_vf_call<ArgmaxImpl<U, false, true, true, true>>(localIndexDst, (__ubuf__ U *)src.GetPhyAddr(), dimA, dimR,
                                                          mainCount, indexBase, actualValidR, localValueDst);
      break;
    }
    for (; dimR % (32U / sizeof(U)) != 0;) {
      uint32_t mainCount = dimR / vlSize;
      asc_vf_call<ArgmaxImpl<U, false, false, true, true>>(localIndexDst, (__ubuf__ U *)src.GetPhyAddr(), dimA, dimR,
                                                           mainCount, indexBase, actualValidR, localValueDst);
      break;
    }
  } else {
    if (dimA >= vlSize || (actualValidR <= vlSize && actualValidR <= 2U * dimA)) {
      asc_vf_call<ArgmaxRaVectorImpl<U, true>>(localIndexDst, (__ubuf__ U *)src.GetPhyAddr(), dimA, actualValidR,
                                               indexBase, localValueDst);
      return;
    }
    asc_vf_call<ArgmaxImpl<U, true, false, true, true>>(localIndexDst, (__ubuf__ U *)src.GetPhyAddr(), dimA, dimR, 0,
                                                        indexBase, actualValidR, localValueDst);
  }
}

/**
 * Phase2 向量化：A 进 lane、标量走 N，沿核维 N 归约 {N, A} 的 value/index。
 * index 内部统一为 int32；WithValue 场景通过可选 valueDst 复用同一份归约结果同时输出 value。
 * 与确定性合并语义一致（值大者胜/值等取最小全局 index）。
 *
 * 核心结构：
 *   - 外层按 step = min(vlSize(value), 64) 分块遍历 A，A 连续进 lane；
 *   - 内层标量循环 n ∈ [1, dimN)，完整 VL 连续 LoadAlign，尾块 masked Gather；
 *   - 每 lane 独立维护一个列的最优 (value, index)，逐行 ArgmaxMergeMasked 更新；
 *   - 尾块用 activeMask 屏蔽无效 lane。
 *
 * value/index 分别按各自位宽限制读取，不需要源 UB tile 末尾预留 VL slack。
 */
// valueDst == nullptr：Argmax 的 index-only 路径。
// valueDst != nullptr：ArgmaxWithValue 复用同一份归约结果，先写 value，再写 index。
template <typename U, bool withValue>
__simd_vf__ inline void ArgMaxMultiRPhase2VectorImpl(__ubuf__ int32_t *indexDst, __ubuf__ U *valueSrc,
                                                     __ubuf__ int32_t *indexSrc, uint32_t dimN, uint32_t dimAValue,
                                                     uint32_t dimAIndex, uint32_t validA,
                                                     __ubuf__ U *valueDst = nullptr) {
  // 内部 index 寄存器与 value 位宽对齐（保证 mask lane 数一致）；index 存储 dtype 仍为 int32。
  using IndexRegT = std::conditional_t<(sizeof(U) == 8), int64_t, int32_t>;
  constexpr uint32_t valueVl = GetVecLen() / sizeof(U);
  constexpr uint32_t indexVl = GetVecLen() / sizeof(IndexRegT);
  constexpr uint32_t step = valueVl < indexVl ? valueVl : indexVl;
  Reg::MaskReg activeMask;
  Reg::RegTensor<U> bestValue, curValue;
  Reg::RegTensor<IndexRegT> bestIndex, curIndex;
  if constexpr (sizeof(U) == 8) {
    Reg::RegTensor<int32_t> bestI32, curI32, src32, zero32, lo, hi;
    Reg::MaskReg indexAllMask = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zero32, 0, indexAllMask);
    for (uint32_t a0 = 0; a0 < validA; a0 += step) {
      uint32_t count = (validA - a0 < step) ? (validA - a0) : step;
      uint32_t maskCount = count;
      activeMask = Reg::UpdateMask<U>(maskCount);
      ArgmaxLoadRow(bestValue, valueSrc + a0, count);
      ArgmaxLoadRow(src32, indexSrc + a0, count);
      Reg::Interleave(lo, hi, src32, zero32);
      bestIndex = (Reg::RegTensor<IndexRegT> &)lo;
      for (uint32_t n = 1; n < dimN; ++n) {
        ArgmaxLoadRow(curValue, valueSrc + n * dimAValue + a0, count);
        ArgmaxLoadRow(src32, indexSrc + n * dimAIndex + a0, count);
        Reg::Interleave(lo, hi, src32, zero32);
        curIndex = (Reg::RegTensor<IndexRegT> &)lo;
        ArgmaxMergeMasked<U, IndexRegT>(bestValue, bestIndex, curValue, curIndex, activeMask);
      }
      if constexpr (withValue) {
        Reg::Store(valueDst + a0, bestValue, count);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
      }
      Reg::DeInterleave(lo, hi, (Reg::RegTensor<int32_t> &)bestIndex, (Reg::RegTensor<int32_t> &)bestIndex);
      Reg::Store(indexDst + a0, lo, count);
    }
    return;
  }
  for (uint32_t a0 = 0; a0 < validA; a0 += step) {
    uint32_t count = (validA - a0 < step) ? (validA - a0) : step;
    uint32_t maskCount = count;
    activeMask = Reg::UpdateMask<U>(maskCount);
    ArgmaxLoadRow(bestValue, valueSrc + 0 * dimAValue + a0, count);
    ArgmaxPhase2LoadIndexRow<IndexRegT>(bestIndex, indexSrc, 0 * dimAIndex + a0, count);
    for (uint32_t n = 1; n < dimN; ++n) {
      ArgmaxLoadRow(curValue, valueSrc + n * dimAValue + a0, count);
      ArgmaxPhase2LoadIndexRow<IndexRegT>(curIndex, indexSrc, n * dimAIndex + a0, count);
      ArgmaxMergeMasked<U, IndexRegT>(bestValue, bestIndex, curValue, curIndex, activeMask);
    }
    // WithValue 复用同一份归约结果：先写 value，再用 VST_VST 隔离，最后写 index。
    if constexpr (withValue) {
      Reg::Store(valueDst + a0, bestValue, count);
      Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
    }
    ArgmaxPhase2StoreIndexRow<IndexRegT>(bestIndex, indexDst, a0, count);
  }
}

/**
 * srcShape = {dimN, dimA_value, dimA_index}；index 为 int32（内部统一）。
 */
template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2ExtendImpl(const LocalTensor<int32_t> &indexDst,
                                                    const LocalTensor<U> &valueSrc,
                                                    const LocalTensor<int32_t> &indexSrc,
                                                    const LocalTensor<uint8_t> &sharedTmpBuffer,
                                                    const uint32_t *srcShape, uint32_t validA = 0U) {
  const uint32_t dimN = srcShape[0];
  const uint32_t dimAValue = srcShape[1];
  const uint32_t dimAIndex = srcShape[2];
  const uint32_t maxValidA = dimAValue < dimAIndex ? dimAValue : dimAIndex;
  const uint32_t actualValidA = (validA == 0U || validA > maxValidA) ? maxValidA : validA;
  asc_vf_call<ArgMaxMultiRPhase2VectorImpl<U, false>>(
      (__ubuf__ int32_t *)indexDst.GetPhyAddr(), (__ubuf__ U *)valueSrc.GetPhyAddr(),
      (__ubuf__ int32_t *)indexSrc.GetPhyAddr(), dimN, dimAValue, dimAIndex, actualValidA, nullptr);
}

/**
 * ArgmaxWithValue R 分核 Phase2：沿核维度 N 归约，同时输出全局 value 与 index。
 */
template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2WithValueExtendImpl(const LocalTensor<U> &valueDst,
                                                             const LocalTensor<int32_t> &indexDst,
                                                             const LocalTensor<U> &valueSrc,
                                                             const LocalTensor<int32_t> &indexSrc,
                                                             const LocalTensor<uint8_t> &sharedTmpBuffer,
                                                             const uint32_t *srcShape, uint32_t validA = 0U) {
  const uint32_t dimN = srcShape[0];
  const uint32_t dimAValue = srcShape[1];
  const uint32_t dimAIndex = srcShape[2];
  const uint32_t maxValidA = dimAValue < dimAIndex ? dimAValue : dimAIndex;
  const uint32_t actualValidA = (validA == 0U || validA > maxValidA) ? maxValidA : validA;
  asc_vf_call<ArgMaxMultiRPhase2VectorImpl<U, true>>((__ubuf__ int32_t *)indexDst.GetPhyAddr(),
                                                     (__ubuf__ U *)valueSrc.GetPhyAddr(),
                                                     (__ubuf__ int32_t *)indexSrc.GetPhyAddr(), dimN, dimAValue,
                                                     dimAIndex, actualValidA, (__ubuf__ U *)valueDst.GetPhyAddr());
}

/**
 * Returns the first maximum index for AR or RA data.
 * index 输出统一为 int32；int32→int64 的转换由 codegen 在最终出口完成。
 * srcInnerPad is reserved for compatibility; pass false for compact input.
 */
template <typename U, class pattern>
__aicore__ inline void ArgMaxExtend(const LocalTensor<int32_t> &dst, const LocalTensor<U> &src,
                                    const uint32_t *srcShape, bool srcInnerPad, uint32_t validR = 0U) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(SupportType<U, float, int32_t>(), "ArgMaxExtend: input type must be float or int32_t");
  static_assert(SupportType<pattern, Pattern::Reduce::AR, Pattern::Reduce::RA>(),
                "ArgMaxExtend: only AR and RA patterns are supported");
  ArgMaxExtendImpl<U, pattern>(dst, src, srcShape, validR);
}

/**
 * 输出局部最大值（valueDst）与其索引（indexDst），无临时 buffer。
 * 供前端 ArgmaxWithValue 单算子（indexBase=0）与 R 轴分核 Phase1（indexBase=核偏移，index 全局化）共用。
 */
template <typename U, class pattern>
__aicore__ inline void ArgMaxWithValueExtend(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &valueDst,
                                             const LocalTensor<U> &src, const LocalTensor<uint8_t> &sharedTmpBuffer,
                                             const uint32_t *srcShape, bool srcInnerPad, uint32_t indexBase = 0U,
                                             uint32_t validR = 0U) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(SupportType<U, float, int32_t, int64_t>(),
                "ArgMaxWithValueExtend: input type must be float, int32_t or int64_t");
  static_assert(SupportType<pattern, Pattern::Reduce::AR, Pattern::Reduce::RA>(),
                "ArgMaxWithValueExtend: only AR and RA patterns are supported");
  ArgMaxWithValueExtendImpl<U, pattern>(indexDst, valueDst, src, sharedTmpBuffer, srcShape, indexBase, validR);
}

template <typename U, class pattern>
__aicore__ inline void ArgMaxWithValueExtend(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &valueDst,
                                             const LocalTensor<U> &src, const uint32_t *srcShape, bool srcInnerPad,
                                             uint32_t indexBase = 0U, uint32_t validR = 0U) {
  LocalTensor<uint8_t> sharedTmpBuffer;
  bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
  ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
  ArgMaxWithValueExtend<U, pattern>(indexDst, valueDst, src, sharedTmpBuffer, srcShape, srcInnerPad, indexBase, validR);
}

/**
 * srcShape = {dimN, dimA_value, dimA_index}；index 为 int32（内部统一）。
 */
template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2Extend(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &valueSrc,
                                                const LocalTensor<int32_t> &indexSrc,
                                                const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t *srcShape,
                                                uint32_t validA = 0U) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(SupportType<U, float, int32_t>(), "ArgMaxMultiRPhase2Extend: value type must be float or int32_t");
  ArgMaxMultiRPhase2ExtendImpl<U>(indexDst, valueSrc, indexSrc, sharedTmpBuffer, srcShape, validA);
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2Extend(const LocalTensor<int32_t> &indexDst, const LocalTensor<U> &valueSrc,
                                                const LocalTensor<int32_t> &indexSrc, const uint32_t *srcShape,
                                                uint32_t validA = 0U) {
  LocalTensor<uint8_t> sharedTmpBuffer;
  bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
  ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
  ArgMaxMultiRPhase2Extend<U>(indexDst, valueSrc, indexSrc, sharedTmpBuffer, srcShape, validA);
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2WithValueExtend(const LocalTensor<U> &valueDst,
                                                         const LocalTensor<int32_t> &indexDst,
                                                         const LocalTensor<U> &valueSrc,
                                                         const LocalTensor<int32_t> &indexSrc,
                                                         const LocalTensor<uint8_t> &sharedTmpBuffer,
                                                         const uint32_t *srcShape, uint32_t validA = 0U) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(SupportType<U, float, int32_t, int64_t>(),
                "ArgMaxMultiRPhase2WithValueExtend: value type must be float, int32_t or int64_t");
  ArgMaxMultiRPhase2WithValueExtendImpl<U>(valueDst, indexDst, valueSrc, indexSrc, sharedTmpBuffer, srcShape, validA);
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2WithValueExtend(const LocalTensor<U> &valueDst,
                                                         const LocalTensor<int32_t> &indexDst,
                                                         const LocalTensor<U> &valueSrc,
                                                         const LocalTensor<int32_t> &indexSrc, const uint32_t *srcShape,
                                                         uint32_t validA = 0U) {
  LocalTensor<uint8_t> sharedTmpBuffer;
  bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
  ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
  ArgMaxMultiRPhase2WithValueExtend<U>(valueDst, indexDst, valueSrc, indexSrc, sharedTmpBuffer, srcShape, validA);
}

/**
 * Phase2 向量化入口：沿核维 N 归约 {N, A}，输出全局 index。
 * srcShape = {dimN, dimA_value, dimA_index}；index 为 int32（内部统一）。
 */
template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2VectorExtend(const LocalTensor<int32_t> &indexDst,
                                                      const LocalTensor<U> &valueSrc,
                                                      const LocalTensor<int32_t> &indexSrc,
                                                      const LocalTensor<uint8_t> &sharedTmpBuffer,
                                                      const uint32_t *srcShape, uint32_t validA = 0U) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(SupportType<U, float, int32_t, int64_t>(),
                "ArgMaxMultiRPhase2VectorExtend: value type must be float, int32_t or int64_t");
  const uint32_t dimN = srcShape[0];
  const uint32_t dimAValue = srcShape[1];
  const uint32_t dimAIndex = srcShape[2];
  const uint32_t maxValidA = dimAValue < dimAIndex ? dimAValue : dimAIndex;
  const uint32_t actualValidA = (validA == 0U || validA > maxValidA) ? maxValidA : validA;
  asc_vf_call<ArgMaxMultiRPhase2VectorImpl<U, false>>(
      (__ubuf__ int32_t *)indexDst.GetPhyAddr(), (__ubuf__ U *)valueSrc.GetPhyAddr(),
      (__ubuf__ int32_t *)indexSrc.GetPhyAddr(), dimN, dimAValue, dimAIndex, actualValidA, nullptr);
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2VectorExtend(const LocalTensor<int32_t> &indexDst,
                                                      const LocalTensor<U> &valueSrc,
                                                      const LocalTensor<int32_t> &indexSrc, const uint32_t *srcShape,
                                                      uint32_t validA = 0U) {
  LocalTensor<uint8_t> sharedTmpBuffer;
  bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
  ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
  ArgMaxMultiRPhase2VectorExtend<U>(indexDst, valueSrc, indexSrc, sharedTmpBuffer, srcShape, validA);
}

/**
 * Phase2 WithValue 向量化入口：沿核维 N 归约，同时输出全局 value 与全局 index。
 * srcShape = {dimN, dimA_value, dimA_index}；index 为 int32（内部统一）。
 */
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
  static_assert(SupportType<U, float, int32_t, int64_t>(),
                "ArgMaxMultiRPhase2WithValueVectorExtend: value type must be float, int32_t or int64_t");
  const uint32_t dimN = srcShape[0];
  const uint32_t dimAValue = srcShape[1];
  const uint32_t dimAIndex = srcShape[2];
  const uint32_t maxValidA = dimAValue < dimAIndex ? dimAValue : dimAIndex;
  const uint32_t actualValidA = (validA == 0U || validA > maxValidA) ? maxValidA : validA;
  asc_vf_call<ArgMaxMultiRPhase2VectorImpl<U, true>>((__ubuf__ int32_t *)indexDst.GetPhyAddr(),
                                                     (__ubuf__ U *)valueSrc.GetPhyAddr(),
                                                     (__ubuf__ int32_t *)indexSrc.GetPhyAddr(), dimN, dimAValue,
                                                     dimAIndex, actualValidA, (__ubuf__ U *)valueDst.GetPhyAddr());
}

template <typename U>
__aicore__ inline void ArgMaxMultiRPhase2WithValueVectorExtend(const LocalTensor<U> &valueDst,
                                                               const LocalTensor<int32_t> &indexDst,
                                                               const LocalTensor<U> &valueSrc,
                                                               const LocalTensor<int32_t> &indexSrc,
                                                               const uint32_t *srcShape, uint32_t validA = 0U) {
  LocalTensor<uint8_t> sharedTmpBuffer;
  bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
  ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
  ArgMaxMultiRPhase2WithValueVectorExtend<U>(valueDst, indexDst, valueSrc, indexSrc, sharedTmpBuffer, srcShape, validA);
}

}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_ARGMAX_H__
