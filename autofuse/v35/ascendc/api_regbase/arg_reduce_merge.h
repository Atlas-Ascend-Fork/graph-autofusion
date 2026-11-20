/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ARG_REDUCE_MERGE_H__
#define __ASCENDC_API_REGBASE_ARG_REDUCE_MERGE_H__

#include <cstdint>
#include <limits>
#include <type_traits>
#ifndef __ASCENDC_API_REGBASE_ARG_REDUCE_COMMON_H__
#include "arg_reduce_common.h"
#endif

namespace AscendC {
// Initialize, materialize and merge partial (value, index) results across R tiles or cores.
template <typename T>
__simd_vf__ inline void ArgReduceCopyImpl(__ubuf__ T *dst, __ubuf__ T *src, uint32_t count) {
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
__aicore__ inline void ArgReduceCopy(const LocalTensor<T> &dst, const LocalTensor<T> &src, uint32_t count) {
  asc_vf_call<ArgReduceCopyImpl<T>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), count);
}

// ArgmaxMergeRound 向量化 vf：value/index 同一轮完成选择并写回。
// value 与 index 的 Store 之间插入 LocalMemBar，保证不同类型输出的写回顺序。
// 块内列数取 min(vlSize(value), 64)，activeMask 只激活 count 个 lane。
template <typename U, bool isMax>
__simd_vf__ inline void ArgReduceMergeRoundImpl(__ubuf__ U *bestValue, __ubuf__ int32_t *bestIndex,
                                                __ubuf__ U *curValue, __ubuf__ int32_t *curIndex, uint32_t dimA) {
  using IndexRegT = std::conditional_t<(sizeof(U) == 8), int64_t, int32_t>;
  constexpr uint32_t valueVl = GetVecLen() / sizeof(U);
  constexpr uint32_t indexVl = GetVecLen() / sizeof(IndexRegT);
  constexpr uint32_t step = valueVl < indexVl ? valueVl : indexVl;
  Reg::RegTensor<U> bestV, curV;
  Reg::RegTensor<IndexRegT> bestI, curI;
  if constexpr (sizeof(U) == 8) {
    Reg::RegTensor<int32_t> bestI32, curI32, zero32, lo, hi;
    Reg::MaskReg indexAllMask = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zero32, 0, indexAllMask);
    for (uint32_t a0 = 0; a0 < dimA; a0 += step) {
      uint32_t count = (dimA - a0 < step) ? (dimA - a0) : step;
      uint32_t maskCount = count;
      Reg::MaskReg mask = Reg::UpdateMask<U>(maskCount);
      ArgReduceLoadRow(bestV, bestValue + a0, count);
      ArgReduceLoadRow(curV, curValue + a0, count);
      ArgReduceLoadRow(bestI32, bestIndex + a0, count);
      ArgReduceLoadRow(curI32, curIndex + a0, count);
      Reg::Interleave(lo, hi, bestI32, zero32);
      bestI = (Reg::RegTensor<IndexRegT> &)lo;
      Reg::Interleave(lo, hi, curI32, zero32);
      curI = (Reg::RegTensor<IndexRegT> &)lo;
      ArgReduceMergeMasked<U, IndexRegT, isMax>(bestV, bestI, curV, curI, mask);
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
    ArgReduceLoadRow(bestV, bestValue + a0, count);
    ArgReduceLoadRow(curV, curValue + a0, count);
    ArgReduceLoadIndexBlock<IndexRegT>(bestI, bestIndex, a0, count);
    ArgReduceLoadIndexBlock<IndexRegT>(curI, curIndex, a0, count);
    ArgReduceMergeMasked<U, IndexRegT, isMax>(bestV, bestI, curV, curI, mask);
    Reg::Store(bestValue + a0, bestV, count);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
    ArgReduceStoreIndexBlock<IndexRegT>(bestI, bestIndex, a0, count);
  }
}

// 增量合并一次 R 轴分块归约结果（供 kCommon 模板 R 轴分块 multi_reduce 使用）：
// 与 Phase2/ArgmaxMerge 的确定性语义对齐：值大者胜；值等且 index 小者胜。
// 同宽/混宽、主块/尾块统一走向量化合并（无 GetValue/SetValue 标量路径）。
template <typename U, bool isMax>
__aicore__ inline void ArgReduceMergeRound(const LocalTensor<U> &bestValue, const LocalTensor<int32_t> &bestIndex,
                                           const LocalTensor<U> &curValue, const LocalTensor<int32_t> &curIndex,
                                           uint32_t dimA) {
  asc_vf_call<ArgReduceMergeRoundImpl<U, isMax>>(
      (__ubuf__ U *)bestValue.GetPhyAddr(), (__ubuf__ int32_t *)bestIndex.GetPhyAddr(),
      (__ubuf__ U *)curValue.GetPhyAddr(), (__ubuf__ int32_t *)curIndex.GetPhyAddr(), dimA);
}

// Phase2 wide-value path: retain one zero-register initialization per VF and int32 index storage.
template <typename U, bool withValue, bool isMax>
__simd_callee__ inline void ArgReducePhase2WideReduce(__ubuf__ int32_t *indexDst, __ubuf__ U *valueSrc,
                                                      __ubuf__ int32_t *indexSrc, uint32_t dimN, uint32_t dimAValue,
                                                      uint32_t dimAIndex, uint32_t validA, __ubuf__ U *valueDst) {
  using IndexRegT = std::conditional_t<(sizeof(U) == 8), int64_t, int32_t>;
  constexpr uint32_t valueVl = GetVecLen() / sizeof(U);
  constexpr uint32_t indexVl = GetVecLen() / sizeof(IndexRegT);
  constexpr uint32_t step = valueVl < indexVl ? valueVl : indexVl;
  Reg::MaskReg activeMask;
  Reg::RegTensor<U> bestValue, curValue;
  Reg::RegTensor<IndexRegT> bestIndex, curIndex;
  Reg::RegTensor<int32_t> src32, zero32, lo, hi;
  Reg::MaskReg indexAllMask = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
  Reg::Duplicate(zero32, 0, indexAllMask);
  for (uint32_t a0 = 0; a0 < validA; a0 += step) {
    uint32_t count = (validA - a0 < step) ? (validA - a0) : step;
    uint32_t maskCount = count;
    activeMask = Reg::UpdateMask<U>(maskCount);
    ArgReduceLoadRow(bestValue, valueSrc + a0, count);
    ArgReduceLoadRow(src32, indexSrc + a0, count);
    Reg::Interleave(lo, hi, src32, zero32);
    bestIndex = (Reg::RegTensor<IndexRegT> &)lo;
    for (uint32_t n = 1; n < dimN; ++n) {
      ArgReduceLoadRow(curValue, valueSrc + n * dimAValue + a0, count);
      ArgReduceLoadRow(src32, indexSrc + n * dimAIndex + a0, count);
      Reg::Interleave(lo, hi, src32, zero32);
      curIndex = (Reg::RegTensor<IndexRegT> &)lo;
      ArgReduceMergeMasked<U, IndexRegT, isMax>(bestValue, bestIndex, curValue, curIndex, activeMask);
    }
    if constexpr (withValue) {
      Reg::Store(valueDst + a0, bestValue, count);
      Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
    }
    Reg::DeInterleave(lo, hi, (Reg::RegTensor<int32_t> &)bestIndex, (Reg::RegTensor<int32_t> &)bestIndex);
    Reg::Store(indexDst + a0, lo, count);
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
template <typename U, bool withValue, bool isMax>
__simd_vf__ inline void ArgReducePhase2VectorImpl(__ubuf__ int32_t *indexDst, __ubuf__ U *valueSrc,
                                                  __ubuf__ int32_t *indexSrc, uint32_t dimN, uint32_t dimAValue,
                                                  uint32_t dimAIndex, uint32_t validA, __ubuf__ U *valueDst = nullptr) {
  if constexpr (sizeof(U) == 8) {
    ArgReducePhase2WideReduce<U, withValue, isMax>(indexDst, valueSrc, indexSrc, dimN, dimAValue, dimAIndex, validA,
                                                   valueDst);
    return;
  }
  // 内部 index 寄存器与 value 位宽对齐（保证 mask lane 数一致）；index 存储 dtype 仍为 int32。
  using IndexRegT = std::conditional_t<(sizeof(U) == 8), int64_t, int32_t>;
  constexpr uint32_t valueVl = GetVecLen() / sizeof(U);
  constexpr uint32_t indexVl = GetVecLen() / sizeof(IndexRegT);
  constexpr uint32_t step = valueVl < indexVl ? valueVl : indexVl;
  Reg::MaskReg activeMask;
  Reg::RegTensor<U> bestValue, curValue;
  Reg::RegTensor<IndexRegT> bestIndex, curIndex;
  for (uint32_t a0 = 0; a0 < validA; a0 += step) {
    uint32_t count = (validA - a0 < step) ? (validA - a0) : step;
    uint32_t maskCount = count;
    activeMask = Reg::UpdateMask<U>(maskCount);
    ArgReduceLoadRow(bestValue, valueSrc + a0, count);
    ArgReduceLoadIndexBlock<IndexRegT>(bestIndex, indexSrc, a0, count);
    for (uint32_t n = 1; n < dimN; ++n) {
      ArgReduceLoadRow(curValue, valueSrc + n * dimAValue + a0, count);
      ArgReduceLoadIndexBlock<IndexRegT>(curIndex, indexSrc, n * dimAIndex + a0, count);
      ArgReduceMergeMasked<U, IndexRegT, isMax>(bestValue, bestIndex, curValue, curIndex, activeMask);
    }
    // WithValue 复用同一份归约结果：先写 value，再用 VST_VST 隔离，最后写 index。
    if constexpr (withValue) {
      Reg::Store(valueDst + a0, bestValue, count);
      Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
    }
    ArgReduceStoreIndexBlock<IndexRegT>(bestIndex, indexDst, a0, count);
  }
}

}  // namespace AscendC

#endif  // __ASCENDC_API_REGBASE_ARG_REDUCE_MERGE_H__
