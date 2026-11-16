/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/argmax.h"
#include "api_regbase/cast.h"

using namespace AscendC;

namespace af {
namespace {
constexpr uint32_t kRows = 6;

constexpr uint32_t AlignBlock(uint32_t bytes) {
  return (bytes + 31U) / 32U * 32U;
}

template <typename T, bool isRa>
void FillInput(T *src, uint32_t columns, bool special = false, uint32_t rows = kRows) {
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t column = 0; column < columns; ++column) {
      T value = static_cast<T>(column);
      if (row == 1) {
        value = static_cast<T>(-3);
      } else if (row == 2) {
        const uint32_t second = columns > 1 ? columns - 2 : 0;
        value = column == std::min(5U, columns - 1) || column == second ? static_cast<T>(99) : static_cast<T>(-100);
      } else if (row == 3) {
        value = static_cast<T>(-static_cast<int64_t>(column));
        const uint32_t first = std::min(5U, columns - 1);
        const uint32_t second = std::max(first, columns > 1 ? columns - 2 : 0);
        if (column == first || column == second) {
          if constexpr (std::is_same_v<T, float>) {
            value = 1.0e30F;
          } else if constexpr (std::is_same_v<T, int64_t>) {
            // 覆盖 int64 大数路径（超过 int32 范围，验证 8B 比较/Gather）
            value = std::numeric_limits<int64_t>::max() / 2;
          } else {
            value = (1 << 24) + 1;
          }
        }
      } else if constexpr (std::is_same_v<T, float>) {
        if (row == 4) {
          value = column % 2 == 0 ? -0.0F : 0.0F;
        } else if (row == 5) {
          value = column == 0 ? 0.0F : -0.0F;
        }
      }
      if constexpr (std::is_same_v<T, float>) {
        if (special) {
          if (row == 0) {
            value = std::numeric_limits<float>::quiet_NaN();
          } else if (row == 1) {
            value = column == 1 || column == columns - 1 ? std::numeric_limits<float>::quiet_NaN()
                                                         : std::numeric_limits<float>::infinity();
          } else if (row == 2) {
            value = -std::numeric_limits<float>::infinity();
          } else if (row == 3) {
            value = column == columns - 1 ? std::numeric_limits<float>::infinity() : 1.0F;
          } else if (row == 4) {
            value = column == columns - 1 ? std::numeric_limits<float>::quiet_NaN() : 1.0F;
          }
        }
      }
      src[isRa ? column * rows + row : row * columns + column] = value;
    }
  }
}

template <typename T>
bool IsLarger(T current, T best) {
  if constexpr (std::is_same_v<T, float>) {
    return !std::isnan(best) && (std::isnan(current) || current > best);
  }
  return current > best;
}

template <typename T, bool isRa>
void RunArgmax(uint32_t columns, bool special = false) {
  T *src = static_cast<T *>(GmAlloc(sizeof(T) * kRows * columns));
  int32_t *index = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * (kRows + 2)));
  std::fill_n(index, kRows + 2, static_cast<int32_t>(-1));
  FillInput<T, isRa>(src, columns, special);

  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuf, indexBuf;
    pipe.InitBuffer(srcBuf, AlignBlock(sizeof(T) * kRows * columns));
    pipe.InitBuffer(indexBuf, AlignBlock(sizeof(int32_t) * (kRows + 2)));
    LocalTensor<T> srcLocal = srcBuf.Get<T>();
    LocalTensor<int32_t> indexLocal = indexBuf.Get<int32_t>();
    GmToUb(srcLocal, src, kRows * columns);
    // Nonzero high bits expose partial int64 stores; guard both ends of dst.
    GmToUb(indexLocal, index, kRows + 2);
    uint32_t srcShape[] = {isRa ? columns : kRows, isRa ? kRows : columns};
    if constexpr (isRa) {
      ArgMaxExtend<T, Pattern::Reduce::RA>(indexLocal[1], srcLocal, srcShape, false);
    } else {
      ArgMaxExtend<T, Pattern::Reduce::AR>(indexLocal[1], srcLocal, srcShape, false);
    }
    UbToGm(index, indexLocal, kRows + 2);
  };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  EXPECT_EQ(index[0], static_cast<int32_t>(-1));
  EXPECT_EQ(index[kRows + 1], static_cast<int32_t>(-1));
  for (uint32_t row = 0; row < kRows; ++row) {
    T best = src[isRa ? row : row * columns];
    uint32_t bestIndex = 0;
    for (uint32_t column = 1; column < columns; ++column) {
      T current = src[isRa ? column * kRows + row : row * columns + column];
      if (IsLarger(current, best)) {
        best = current;
        bestIndex = column;
      }
    }
    EXPECT_EQ(index[row + 1], static_cast<int32_t>(bestIndex));
  }
  GmFree(src);
  GmFree(index);
}

// ArgMaxWithValueExtend（前端，1 入 2 出 value+index）：验证 value 为最大值、index 为首个最大位置。
// int64 value 走 ArgMaxWithValueExtend（static_assert 已放开 int64），非 ArgMaxExtend。
template <typename T, bool isRa>
void RunArgmaxWithValue(uint32_t columns, bool special = false, uint32_t validR = 0, uint32_t rows = kRows,
                        uint32_t indexBase = 0) {
  const uint32_t validColumns = validR == 0 ? columns : validR;
  T *src = static_cast<T *>(GmAlloc(sizeof(T) * rows * columns));
  int32_t *index = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * (rows + 2)));
  T *value = static_cast<T *>(GmAlloc(sizeof(T) * (rows + 2)));
  std::fill_n(index, rows + 2, static_cast<int32_t>(-1));
  std::fill_n(value, rows + 2, static_cast<T>(-1));
  FillInput<T, isRa>(src, columns, special, rows);

  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuf, indexBuf, valueBuf;
    pipe.InitBuffer(srcBuf, AlignBlock(sizeof(T) * rows * columns));
    pipe.InitBuffer(indexBuf, AlignBlock(sizeof(int32_t) * (rows + 2)));
    pipe.InitBuffer(valueBuf, AlignBlock(sizeof(T) * (rows + 2)));
    LocalTensor<T> srcLocal = srcBuf.Get<T>();
    LocalTensor<int32_t> indexLocal = indexBuf.Get<int32_t>();
    LocalTensor<T> valueLocal = valueBuf.Get<T>();
    GmToUb(srcLocal, src, rows * columns);
    GmToUb(indexLocal, index, rows + 2);
    GmToUb(valueLocal, value, rows + 2);
    uint32_t srcShape[] = {isRa ? columns : rows, isRa ? rows : columns};
    if constexpr (isRa) {
      ArgMaxWithValueExtend<T, Pattern::Reduce::RA>(indexLocal[1], valueLocal[1], srcLocal, LocalTensor<uint8_t>(),
                                                    srcShape, false, indexBase, validR);
    } else {
      ArgMaxWithValueExtend<T, Pattern::Reduce::AR>(indexLocal[1], valueLocal[1], srcLocal, LocalTensor<uint8_t>(),
                                                    srcShape, false, indexBase, validR);
    }
    UbToGm(index, indexLocal, rows + 2);
    UbToGm(value, valueLocal, rows + 2);
  };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  EXPECT_EQ(index[0], static_cast<int32_t>(-1));
  EXPECT_EQ(index[rows + 1], static_cast<int32_t>(-1));
  EXPECT_EQ(value[0], static_cast<T>(-1));
  EXPECT_EQ(value[rows + 1], static_cast<T>(-1));
  for (uint32_t row = 0; row < rows; ++row) {
    T best = src[isRa ? row : row * columns];
    uint32_t bestIndex = 0;
    for (uint32_t column = 1; column < validColumns; ++column) {
      T current = src[isRa ? column * rows + row : row * columns + column];
      if (IsLarger(current, best)) {
        best = current;
        bestIndex = column;
      }
    }
    if constexpr (std::is_same_v<T, float>) {
      if (std::isnan(best)) {
        EXPECT_TRUE(std::isnan(value[row + 1]));
      } else {
        EXPECT_EQ(value[row + 1], best);
        if (best == 0.0F) {
          EXPECT_EQ(std::signbit(value[row + 1]), std::signbit(best))
              << "row=" << row << ", columns=" << columns << ", isRa=" << isRa;
        }
      }
    } else {
      EXPECT_EQ(value[row + 1], best);
    }
    EXPECT_EQ(index[row + 1], static_cast<int32_t>(bestIndex + indexBase));
  }
  GmFree(src);
  GmFree(index);
  GmFree(value);
}

template <bool isRa>
void RunWithValueTypeMatrix() {
  RunArgmaxWithValue<float, isRa>(67);
  RunArgmaxWithValue<int32_t, isRa>(67);
  RunArgmaxWithValue<int64_t, isRa>(67);
}

template <bool isRa>
void RunWithValueBoundaries() {
  for (uint32_t columns : {1U, 64U, 65U, 136U, 256U, 4096U}) {
    RunArgmaxWithValue<float, isRa>(columns);
    RunArgmaxWithValue<int32_t, isRa>(columns);
    RunArgmaxWithValue<int64_t, isRa>(columns);
  }
}

template <bool isRa>
void RunTypeMatrix() {
  RunArgmax<float, isRa>(67);
  RunArgmax<int32_t, isRa>(67);
}

template <bool isRa>
void RunBoundaries() {
  // Scalar, full VL, unaligned tail, aligned tail, paired repeats, and large input.
  for (uint32_t columns : {1U, 64U, 65U, 136U, 256U, 4096U}) {
    RunArgmax<float, isRa>(columns);
    RunArgmax<int32_t, isRa>(columns);
  }
}

// ArgMaxMultiRPhase2WithValueExtend：沿核维度 N 一次归约同时输出全局 value 与 index。
// 数据设计：每列 a 核0 value=a+1；核1 value=a+2（更大）；核2 value=a+2（与核1并列）；
// 并列取最小全局 index（核1胜）。
template <typename T>
void RunPhase2WithValue(uint32_t dimN, uint32_t dimA, bool special = false) {
  // Phase2 源行跨度 32B 对齐；只分配 tensor 大小，不额外预留 VL slack。
  // 与 RunPhase2VectorWithValue 保持同一契约：按各自 dtype 计算对齐行跨度。
  const uint32_t valueStep = 32U / sizeof(T);
  const uint32_t indexStep = 32U / sizeof(int32_t);
  const uint32_t dimAValue = (dimA + valueStep - 1U) / valueStep * valueStep;
  const uint32_t dimAIndex = (dimA + indexStep - 1U) / indexStep * indexStep;
  T *valueSrc = static_cast<T *>(GmAlloc(sizeof(T) * dimN * dimAValue));
  int32_t *indexSrc = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimN * dimAIndex));
  T *valueDst = static_cast<T *>(GmAlloc(sizeof(T) * dimAValue));
  int32_t *indexDst = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimAIndex));
  std::fill_n(valueDst, dimAValue, static_cast<T>(-1));
  std::fill_n(indexDst, dimAIndex, static_cast<int32_t>(-1));

  const T padValue = std::is_same_v<T, float> ? static_cast<T>(1.0e30F) : std::numeric_limits<T>::max();
  for (uint32_t n = 0; n < dimN; ++n) {
    for (uint32_t a = 0; a < dimAValue; ++a) {
      valueSrc[n * dimAValue + a] = padValue;
    }
    for (uint32_t a = 0; a < dimAIndex; ++a) {
      indexSrc[n * dimAIndex + a] = static_cast<int32_t>(-1000);
    }
  }

  for (uint32_t a = 0; a < dimA; ++a) {
    if constexpr (std::is_same_v<T, int64_t>) {
      // 覆盖 INT64 边界：列0 核0=INT64_MIN、核1=INT64_MAX（胜）、核2=INT64_MAX（并列取最小 index）；
      // 列1 核0=INT64_MAX（胜）、核1=INT64_MIN、核2=-1；其余列常规递增值
      if (a == 0) {
        valueSrc[0 * dimAValue + a] = std::numeric_limits<int64_t>::min();
        valueSrc[1 * dimAValue + a] = std::numeric_limits<int64_t>::max();
        valueSrc[2 * dimAValue + a] = std::numeric_limits<int64_t>::max();
      } else if (a == 1) {
        valueSrc[0 * dimAValue + a] = std::numeric_limits<int64_t>::max();
        valueSrc[1 * dimAValue + a] = std::numeric_limits<int64_t>::min();
        valueSrc[2 * dimAValue + a] = static_cast<T>(-1);
      } else {
        valueSrc[0 * dimAValue + a] = static_cast<T>(a + 1);
        valueSrc[1 * dimAValue + a] = static_cast<T>(a + 2);
        valueSrc[2 * dimAValue + a] = static_cast<T>(a + 2);
      }
    } else {
      valueSrc[0 * dimAValue + a] = static_cast<T>(a + 1);
      valueSrc[1 * dimAValue + a] = static_cast<T>(a + 2);
      valueSrc[2 * dimAValue + a] = static_cast<T>(a + 2);
    }
    for (uint32_t n = 0; n < dimN; ++n) {
      indexSrc[n * dimAIndex + a] = static_cast<int32_t>(n * dimA + a);
    }
  }

  if constexpr (std::is_same_v<T, float>) {
    if (special) {
      for (uint32_t a = 0; a < dimA; ++a) {
        valueSrc[a] = a % 3 == 0 ? std::numeric_limits<float>::quiet_NaN() : 1.0F;
        valueSrc[dimAValue + a] = a % 3 == 1 ? 2.0F : std::numeric_limits<float>::quiet_NaN();
        valueSrc[2 * dimAValue + a] = std::numeric_limits<float>::quiet_NaN();
        // Later cores can carry earlier global indices: NaN ties must compare indices.
        indexSrc[a] = 200 + a;
        indexSrc[dimAIndex + a] = 100 + a;
        indexSrc[2 * dimAIndex + a] = a;
      }
    }
  }

  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> valueSrcBuf, indexSrcBuf, valueDstBuf, indexDstBuf;
    pipe.InitBuffer(valueSrcBuf, AlignBlock(sizeof(T) * dimN * dimAValue));
    pipe.InitBuffer(indexSrcBuf, AlignBlock(sizeof(int32_t) * dimN * dimAIndex));
    pipe.InitBuffer(valueDstBuf, AlignBlock(sizeof(T) * dimAValue));
    pipe.InitBuffer(indexDstBuf, AlignBlock(sizeof(int32_t) * dimAIndex));
    LocalTensor<T> valueSrcLocal = valueSrcBuf.Get<T>();
    LocalTensor<int32_t> indexSrcLocal = indexSrcBuf.Get<int32_t>();
    LocalTensor<T> valueDstLocal = valueDstBuf.Get<T>();
    LocalTensor<int32_t> indexDstLocal = indexDstBuf.Get<int32_t>();
    GmToUb(valueSrcLocal, valueSrc, dimN * dimAValue);
    GmToUb(indexSrcLocal, indexSrc, dimN * dimAIndex);
    GmToUb(valueDstLocal, valueDst, dimAValue);
    GmToUb(indexDstLocal, indexDst, dimAIndex);
    // value / index 按各自 dtype 的 32B 对齐行跨度寻址
    uint32_t srcShape[] = {dimN, dimAValue, dimAIndex};
    ArgMaxMultiRPhase2WithValueExtend<T>(valueDstLocal, indexDstLocal, valueSrcLocal, indexSrcLocal,
                                         LocalTensor<uint8_t>(), srcShape);
    UbToGm(valueDst, valueDstLocal, dimAValue);
    UbToGm(indexDst, indexDstLocal, dimAIndex);
  };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);

  // 独立参考：值大者胜；值等取最小全局 index
  for (uint32_t a = 0; a < dimA; ++a) {
    T best = valueSrc[a];
    int32_t bestIndex = indexSrc[a];
    for (uint32_t n = 1; n < dimN; ++n) {
      T current = valueSrc[n * dimAValue + a];
      int32_t currentIndex = indexSrc[n * dimAIndex + a];
      bool equal = current == best;
      if constexpr (std::is_same_v<T, float>) {
        equal = equal || (std::isnan(current) && std::isnan(best));
      }
      if (IsLarger(current, best) || (equal && currentIndex < bestIndex)) {
        best = current;
        bestIndex = currentIndex;
      }
    }
    if constexpr (std::is_same_v<T, float>) {
      if (std::isnan(best)) {
        EXPECT_TRUE(std::isnan(valueDst[a])) << "a=" << a;
      } else {
        EXPECT_EQ(valueDst[a], best) << "a=" << a;
      }
    } else {
      EXPECT_EQ(valueDst[a], best) << "A=" << dimA << " a=" << a;
    }
    EXPECT_EQ(indexDst[a], bestIndex) << "A=" << dimA << " a=" << a;
  }

  GmFree(valueSrc);
  GmFree(indexSrc);
  GmFree(valueDst);
  GmFree(indexDst);
}

// Phase2 双输入 dtype 不同（如 value 8B / index int32 4B）→ 末轴 32B 对齐后的行跨度不同：
//   value 行跨度 = SizeAlign(A, 32/sizeof(T))，index 行跨度 = SizeAlign(A, 32/sizeof(int32_t))。
// 回归点：Phase2 必须按各自 dtype 的行跨度寻址（此前用同一个 dimA，A%8!=0 时 index 读错行，
// 表现为跨核归约取到错误/越界的 index）。此处把两个输入按不同跨度排布，并把 padding 列填成
// “若被误读必然胜出”的哨兵，保证错误寻址会被本用例暴露。
template <typename T>
void RunPhase2DualPitch(uint32_t dimN, uint32_t aSize) {
  const uint32_t valueStep = 32U / sizeof(T);
  const uint32_t indexStep = 32U / sizeof(int32_t);
  const uint32_t dimAValue = (aSize + valueStep - 1U) / valueStep * valueStep;
  const uint32_t dimAIndex = (aSize + indexStep - 1U) / indexStep * indexStep;
  T *valueSrc = static_cast<T *>(GmAlloc(sizeof(T) * dimN * dimAValue));
  int32_t *indexSrc = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimN * dimAIndex));
  T *valueDst = static_cast<T *>(GmAlloc(sizeof(T) * dimAValue));
  int32_t *indexDst = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimAIndex));
  std::fill_n(valueDst, dimAValue, static_cast<T>(-1));
  std::fill_n(indexDst, dimAIndex, static_cast<int32_t>(-1));

  const T padValue = std::is_same_v<T, float> ? static_cast<T>(1.0e30F) : std::numeric_limits<T>::max();
  for (uint32_t n = 0; n < dimN; ++n) {
    for (uint32_t a = 0; a < dimAValue; ++a) {
      valueSrc[n * dimAValue + a] = padValue;
    }
    for (uint32_t a = 0; a < dimAIndex; ++a) {
      indexSrc[n * dimAIndex + a] = static_cast<int32_t>(-1000);
    }
  }
  // 真实列 a in [0, aSize)：值随 n 递增，第 (a%dimN) 行为该列最大；index 用全局编码区分核。
  for (uint32_t a = 0; a < aSize; ++a) {
    for (uint32_t n = 0; n < dimN; ++n) {
      valueSrc[n * dimAValue + a] = static_cast<T>(n + 1);
      indexSrc[n * dimAIndex + a] = static_cast<int32_t>(n * 1000 + a);
    }
    valueSrc[(a % dimN) * dimAValue + a] = static_cast<T>(dimN + 1);
  }

  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> vSrcBuf, iSrcBuf, vDstBuf, iDstBuf;
    pipe.InitBuffer(vSrcBuf, AlignBlock(sizeof(T) * dimN * dimAValue));
    pipe.InitBuffer(iSrcBuf, AlignBlock(sizeof(int32_t) * dimN * dimAIndex));
    pipe.InitBuffer(vDstBuf, AlignBlock(sizeof(T) * dimAValue));
    pipe.InitBuffer(iDstBuf, AlignBlock(sizeof(int32_t) * dimAIndex));
    LocalTensor<T> vSrcLocal = vSrcBuf.Get<T>();
    LocalTensor<int32_t> iSrcLocal = iSrcBuf.Get<int32_t>();
    LocalTensor<T> vDstLocal = vDstBuf.Get<T>();
    LocalTensor<int32_t> iDstLocal = iDstBuf.Get<int32_t>();
    GmToUb(vSrcLocal, valueSrc, dimN * dimAValue);
    GmToUb(iSrcLocal, indexSrc, dimN * dimAIndex);
    uint32_t srcShape[] = {dimN, dimAValue, dimAIndex};
    ArgMaxMultiRPhase2WithValueExtend<T>(vDstLocal, iDstLocal, vSrcLocal, iSrcLocal, LocalTensor<uint8_t>(), srcShape);
    UbToGm(valueDst, vDstLocal, dimAValue);
    UbToGm(indexDst, iDstLocal, dimAIndex);
  };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  for (uint32_t a = 0; a < aSize; ++a) {
    T best = valueSrc[a];
    int32_t bestIndex = indexSrc[a];
    for (uint32_t n = 1; n < dimN; ++n) {
      const T cur = valueSrc[n * dimAValue + a];
      const int32_t curIndex = indexSrc[n * dimAIndex + a];
      if (cur > best || (cur == best && curIndex < bestIndex)) {
        best = cur;
        bestIndex = curIndex;
      }
    }
    EXPECT_EQ(indexDst[a], bestIndex) << "A=" << aSize << " a=" << a;
  }
  // 同布局再验证单输出变体（ArgMaxMultiRPhase2Extend）；其 value 仅支持 float/int32
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t>) {
    auto kernelSingle = [=] {
      TPipe pipe;
      TBuf<TPosition::VECCALC> vSrcBuf, iSrcBuf, iDstBuf;
      pipe.InitBuffer(vSrcBuf, AlignBlock(sizeof(T) * dimN * dimAValue));
      pipe.InitBuffer(iSrcBuf, AlignBlock(sizeof(int32_t) * dimN * dimAIndex));
      pipe.InitBuffer(iDstBuf, AlignBlock(sizeof(int32_t) * dimAIndex));
      LocalTensor<T> vSrcLocal = vSrcBuf.Get<T>();
      LocalTensor<int32_t> iSrcLocal = iSrcBuf.Get<int32_t>();
      LocalTensor<int32_t> iDstLocal = iDstBuf.Get<int32_t>();
      GmToUb(vSrcLocal, valueSrc, dimN * dimAValue);
      GmToUb(iSrcLocal, indexSrc, dimN * dimAIndex);
      uint32_t srcShape[] = {dimN, dimAValue, dimAIndex};
      ArgMaxMultiRPhase2Extend<T>(iDstLocal, vSrcLocal, iSrcLocal, srcShape);
      UbToGm(indexDst, iDstLocal, dimAIndex);
    };
    std::fill_n(indexDst, dimAIndex, static_cast<int32_t>(-1));
    ICPU_RUN_KF(kernelSingle, 1);
    for (uint32_t a = 0; a < aSize; ++a) {
      T best = valueSrc[a];
      int32_t bestIndex = indexSrc[a];
      for (uint32_t n = 1; n < dimN; ++n) {
        const T cur = valueSrc[n * dimAValue + a];
        const int32_t curIndex = indexSrc[n * dimAIndex + a];
        if (cur > best || (cur == best && curIndex < bestIndex)) {
          best = cur;
          bestIndex = curIndex;
        }
      }
      EXPECT_EQ(indexDst[a], bestIndex) << "single A=" << aSize << " a=" << a;
    }
  }

  GmFree(valueSrc);
  GmFree(indexSrc);
  GmFree(valueDst);
  GmFree(indexDst);
}

// ArgmaxMergeRound：R 轴分块时在历史最佳与当前 tile 结果间做原地合并。
// 关键回归点：codegen 生成的调用以前两个实参为 LocalTensor::operator[] 的右值
// （valueSaved[0]/indexSaved[0]），形参必须按 const 引用接收，否则右值无法绑定
// 非 const 左值引用导致生成 kernel 编译失败。
// 语义：值大者胜；值等且 index 小者胜。
template <typename T>
void RunArgmaxMergeRound(uint32_t dimA, bool special = false) {
  T *valueSaved = static_cast<T *>(GmAlloc(sizeof(T) * dimA));
  int32_t *indexSaved = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimA));
  T *valueCur = static_cast<T *>(GmAlloc(sizeof(T) * dimA));
  int32_t *indexCur = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimA));

  for (uint32_t a = 0; a < dimA; ++a) {
    valueSaved[a] = static_cast<T>(a + 1);
    indexSaved[a] = static_cast<int32_t>(a);
    valueCur[a] = static_cast<T>(a);
    indexCur[a] = static_cast<int32_t>(a + 1);
    // Repeat value/index update patterns across full vectors and the final partial vector.
    switch (a % 6) {
      case 0:
      case 3:
        valueCur[a] = static_cast<T>(a + 100);
        break;
      case 1:
        valueCur[a] = valueSaved[a];
        indexCur[a] = 0;
        break;
      case 2:
        valueCur[a] = valueSaved[a];
        break;
      case 4:
        valueCur[a] = static_cast<T>(-1);
        break;
      default:
        valueCur[a] = valueSaved[a];
        indexCur[a] = indexSaved[a];
        break;
    }
  }

  if constexpr (std::is_same_v<T, float>) {
    if (special) {
      for (uint32_t a = 0; a < dimA; ++a) {
        valueSaved[a] = a % 3 == 0 ? 1.0F : std::numeric_limits<float>::quiet_NaN();
        valueCur[a] = a % 3 == 1 ? 2.0F : std::numeric_limits<float>::quiet_NaN();
        indexSaved[a] = 100 + a;
        indexCur[a] = a;
      }
    }
  }

  // 独立参考：在 kernel 运行（原地写回 GM）前基于输入计算期望
  std::vector<T> expectedValue(dimA);
  std::vector<int32_t> expectedIndex(dimA);
  for (uint32_t a = 0; a < dimA; ++a) {
    T bestValue = valueSaved[a];
    int32_t bestIndex = indexSaved[a];
    const T curValue = valueCur[a];
    const int32_t curIndex = indexCur[a];
    bool equal = curValue == bestValue;
    if constexpr (std::is_same_v<T, float>) {
      equal = equal || (std::isnan(curValue) && std::isnan(bestValue));
    }
    bool update = IsLarger(curValue, bestValue) || (equal && curIndex < bestIndex);
    expectedValue[a] = update ? curValue : bestValue;
    expectedIndex[a] = update ? curIndex : bestIndex;
  }

  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> valueSavedBuf, indexSavedBuf, valueCurBuf, indexCurBuf;
    pipe.InitBuffer(valueSavedBuf, AlignBlock(sizeof(T) * dimA));
    pipe.InitBuffer(indexSavedBuf, AlignBlock(sizeof(int32_t) * dimA));
    pipe.InitBuffer(valueCurBuf, AlignBlock(sizeof(T) * dimA));
    pipe.InitBuffer(indexCurBuf, AlignBlock(sizeof(int32_t) * dimA));
    LocalTensor<T> valueSavedLocal = valueSavedBuf.Get<T>();
    LocalTensor<int32_t> indexSavedLocal = indexSavedBuf.Get<int32_t>();
    LocalTensor<T> valueCurLocal = valueCurBuf.Get<T>();
    LocalTensor<int32_t> indexCurLocal = indexCurBuf.Get<int32_t>();
    GmToUb(valueSavedLocal, valueSaved, dimA);
    GmToUb(indexSavedLocal, indexSaved, dimA);
    GmToUb(valueCurLocal, valueCur, dimA);
    GmToUb(indexCurLocal, indexCur, dimA);
    // 与 codegen 生成调用一致：前两个实参为 operator[] 右值
    ArgmaxMergeRound<T>(valueSavedLocal[0], indexSavedLocal[0], valueCurLocal[0], indexCurLocal[0], dimA);
    UbToGm(valueSaved, valueSavedLocal, dimA);
    UbToGm(indexSaved, indexSavedLocal, dimA);
  };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  for (uint32_t a = 0; a < dimA; ++a) {
    if constexpr (std::is_same_v<T, float>) {
      if (std::isnan(expectedValue[a])) {
        EXPECT_TRUE(std::isnan(valueSaved[a])) << "a=" << a;
      } else {
        EXPECT_EQ(valueSaved[a], expectedValue[a]) << "a=" << a;
      }
    } else {
      EXPECT_EQ(valueSaved[a], expectedValue[a]) << "a=" << a;
    }
    EXPECT_EQ(indexSaved[a], expectedIndex[a]) << "a=" << a;
  }

  GmFree(valueSaved);
  GmFree(indexSaved);
  GmFree(valueCur);
  GmFree(indexCur);
}

// Replay only the guarded final-R-tile API pipeline, not generated code or backend E2E.
template <typename IndexT>
void RunArgmaxFinalRoundOutput(uint32_t columns) {
  constexpr uint32_t tileA = 64;  // int32 -> int64 CastExtend reads a full source VL.
  constexpr uint32_t tileR = 64;
  constexpr uint32_t outerTiles = 2;
  constexpr uint32_t rows = tileA * outerTiles;
  constexpr IndexT sentinel = static_cast<IndexT>(-1234567);
  const uint32_t rounds = (columns + tileR - 1) / tileR;
  auto *src = static_cast<float *>(GmAlloc(sizeof(float) * rows * columns));
  auto *value = static_cast<float *>(GmAlloc(sizeof(float) * rows));
  auto *index = static_cast<IndexT *>(GmAlloc(sizeof(IndexT) * rows));
  auto *history = static_cast<IndexT *>(GmAlloc(sizeof(IndexT) * rows * rounds));
  auto *materializations = static_cast<uint32_t *>(GmAlloc(sizeof(uint32_t) * outerTiles));
  std::vector<float> expectedValue(rows);
  std::vector<IndexT> expectedIndex(rows);

  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuf, curValueBuf, curIndexBuf, savedValueBuf, savedIndexBuf, outputBuf;
    pipe.InitBuffer(srcBuf, sizeof(float) * tileA * tileR);
    pipe.InitBuffer(curValueBuf, sizeof(float) * tileA);
    pipe.InitBuffer(curIndexBuf, sizeof(int32_t) * tileA);
    pipe.InitBuffer(savedValueBuf, sizeof(float) * tileA);
    pipe.InitBuffer(savedIndexBuf, sizeof(int32_t) * tileA);
    pipe.InitBuffer(outputBuf, sizeof(IndexT) * tileA);
    auto localSrc = srcBuf.Get<float>();
    auto curValue = curValueBuf.Get<float>();
    auto curIndex = curIndexBuf.Get<int32_t>();
    auto savedValue = savedValueBuf.Get<float>();
    auto savedIndex = savedIndexBuf.Get<int32_t>();
    auto output = outputBuf.Get<IndexT>();
    uint32_t shape[] = {tileA, tileR};
    uint32_t castShape[] = {tileA};
    uint32_t stride[] = {1};
    for (uint32_t aTile = 0; aTile < outerTiles; ++aTile) {
      const uint32_t aBase = aTile * tileA;
      GmToUb(output, index + aBase, tileA);
      for (uint32_t rBase = 0; rBase < columns; rBase += tileR) {
        const uint32_t validR = std::min(tileR, columns - rBase);
        for (uint32_t a = 0; a < tileA; ++a) {
          auto localRow = localSrc[a * tileR];
          GmToUb(localRow, src + (aBase + a) * columns + rBase, validR);
          for (uint32_t r = validR; r < tileR; ++r) {
            localSrc.SetValue(a * tileR + r, 1.0e6F);  // Poison padding excluded by validR.
          }
        }
        ArgMaxWithValueExtend<float, Pattern::Reduce::AR>(curIndex, curValue, localSrc, LocalTensor<uint8_t>(), shape,
                                                          false, rBase, validR);
        if (rBase == 0) {
          ArgmaxCopy(savedValue[0], curValue[0], tileA);
          ArgmaxCopy(savedIndex[0], curIndex[0], tileA);
        } else {
          ArgmaxMergeRound<float>(savedValue[0], savedIndex[0], curValue[0], curIndex[0], tileA);
        }
        if (rBase + tileR >= columns) {
          if constexpr (std::is_same_v<IndexT, int64_t>) {
            CastExtend<int32_t, int64_t, 1>(output[0], savedIndex[0], castShape, stride, stride);
          } else {
            ArgmaxCopy(output[0], savedIndex[0], tileA);
          }
          ++materializations[aTile];
        }
        UbToGm(history + (aTile * rounds + rBase / tileR) * tileA, output, tileA);
      }
      UbToGm(value + aBase, savedValue, tileA);
      UbToGm(index + aBase, output, tileA);
    }
  };

  SetKernelMode(KernelMode::AIV_MODE);
  for (uint32_t launch = 0; launch < 2; ++launch) {
    std::fill_n(index, rows, sentinel);
    std::fill_n(value, rows, -1.0F);
    std::fill_n(history, rows * rounds, sentinel);
    std::fill_n(materializations, outerTiles, 0U);
    for (uint32_t row = 0; row < rows; ++row) {
      float *input = src + row * columns;
      std::fill_n(input, columns, -3.0F);
      // Rotate winners across launches and A tiles; ties span the first and last R tiles.
      const uint32_t scenario = (row + row / tileA + launch) % 4;
      const uint32_t winner = scenario == 0 || scenario == 3 ? 1U : scenario == 1 ? columns / 2 : columns - 1;
      input[winner] = static_cast<float>(1000 + row + launch * rows);
      if (scenario == 3) {
        input[columns - 1] = input[winner];
      }
      // Independent host golden scans the entire logical R axis, preserving the first tie.
      expectedValue[row] = input[0];
      expectedIndex[row] = 0;
      for (uint32_t r = 1; r < columns; ++r) {
        if (input[r] > expectedValue[row]) {
          expectedValue[row] = input[r];
          expectedIndex[row] = static_cast<IndexT>(r);
        }
      }
    }
    ICPU_RUN_KF(kernel, 1);
    for (uint32_t aTile = 0; aTile < outerTiles; ++aTile) {
      EXPECT_EQ(materializations[aTile], 1U) << "launch=" << launch << " A tile=" << aTile;
      for (uint32_t a = 0; a < tileA; ++a) {
        const uint32_t row = aTile * tileA + a;
        EXPECT_EQ(value[row], expectedValue[row]) << "launch=" << launch << " row=" << row;
        EXPECT_EQ(index[row], expectedIndex[row]) << "launch=" << launch << " row=" << row;
        for (uint32_t round = 0; round < rounds; ++round) {
          EXPECT_EQ(history[(aTile * rounds + round) * tileA + a], round + 1 == rounds ? expectedIndex[row] : sentinel)
              << "launch=" << launch << " row=" << row << " round=" << round;
        }
      }
    }
  }
  GmFree(src);
  GmFree(value);
  GmFree(index);
  GmFree(history);
  GmFree(materializations);
}
}  // namespace

TEST(TestApiArgmax, MultiReduceFinalRoundIndexOutput) {
  for (uint32_t columns : {64U, 192U, 145U}) {
    RunArgmaxFinalRoundOutput<int32_t>(columns);
    RunArgmaxFinalRoundOutput<int64_t>(columns);
  }
}

TEST(TestApiArgmax, ARTypeMatrix) {
  RunTypeMatrix<false>();
}

TEST(TestApiArgmax, NaNRowReduction) {
  for (uint32_t columns : {67U, 136U}) {
    RunArgmax<float, false>(columns, true);
    RunArgmax<float, true>(columns, true);
    RunArgmaxWithValue<float, false>(columns, true);
    RunArgmaxWithValue<float, true>(columns, true);
    // The last physical element is NaN on row 4 but outside validR.
    RunArgmaxWithValue<float, false>(columns, true, columns - 1);
  }
}

TEST(TestApiArgmax, NaNAcrossTilesAndCores) {
  RunArgmaxMergeRound<float>(65, true);
  RunPhase2WithValue<float>(3, 65, true);
}

// 前端 ArgMaxWithValueExtend 的 int64 支持（FR13）：AR/RA × float/int32/int64 值类型
TEST(TestApiArgmax, WithValueARTypeMatrix) {
  RunWithValueTypeMatrix<false>();
}

TEST(TestApiArgmax, WithValueRATypeMatrix) {
  RunWithValueTypeMatrix<true>();
}

TEST(TestApiArgmax, RAVectorRowsAndValidR) {
  // Target inner A=16, full vectors, unaligned row pitches, and multi-vector tails.
  for (uint32_t rows : {16U, 32U, 64U, 65U, 136U}) {
    RunArgmaxWithValue<float, true>(32, false, 0, rows, 97);
    RunArgmaxWithValue<int32_t, true>(32, false, 0, rows, 97);
    RunArgmaxWithValue<int64_t, true>(32, false, 0, rows, 97);
    RunArgmaxWithValue<float, true>(32, true, 31, rows, 97);
    RunArgmaxWithValue<float, true>(137, false, 0, rows, 97);
  }
}

TEST(TestApiArgmax, WithValueARBoundaries) {
  RunWithValueBoundaries<false>();
}

TEST(TestApiArgmax, WithValueRABoundaries) {
  RunWithValueBoundaries<true>();
}

TEST(TestApiArgmax, RATypeMatrix) {
  RunTypeMatrix<true>();
}

TEST(TestApiArgmax, ARBoundaries) {
  RunBoundaries<false>();
}

TEST(TestApiArgmax, RABoundaries) {
  RunBoundaries<true>();
}

TEST(TestApiArgmax, WithValueARValidRBoundaries) {
  // A larger physical stride must not admit padding into full-block loads or the tail.
  for (uint32_t validR : {1U, 63U, 64U, 65U, 191U, 192U, 193U, 256U}) {
    RunArgmaxWithValue<float, false>(256, false, validR);
    RunArgmaxWithValue<int32_t, false>(256, false, validR);
    RunArgmaxWithValue<int64_t, false>(256, false, validR);
  }
}

TEST(TestApiArgmax, RCore129280GlobalIndex) {
  constexpr uint32_t dimR = 129280;
  constexpr uint32_t tileR = 2020;
  constexpr uint32_t dimN = dimR / tileR;
  constexpr uint32_t pitchR = AlignBlock(tileR * sizeof(float)) / sizeof(float);
  constexpr uint32_t phase2Pitch = 8;
  auto *src = static_cast<float *>(GmAlloc(dimR * sizeof(float)));
  auto *index = static_cast<int32_t *>(GmAlloc(sizeof(int32_t)));
  for (uint32_t scenario = 0; scenario < 2; ++scenario) {
    std::fill_n(src, dimR, -3.0F);
    // Last element wins; a cross-partition tie selects the first.
    src[dimR - 1] = -1.0F;
    if (scenario == 1) {
      src[1] = -1.0F;
    }
    index[0] = -1;
    auto kernel = [=] {
      TPipe pipe;
      TBuf<TPosition::VECCALC> srcBuf, valueBuf, indexBuf, tmpValueBuf, tmpIndexBuf, dstBuf;
      pipe.InitBuffer(srcBuf, pitchR * sizeof(float));
      pipe.InitBuffer(valueBuf, dimN * phase2Pitch * sizeof(float));
      pipe.InitBuffer(indexBuf, dimN * phase2Pitch * sizeof(int32_t));
      pipe.InitBuffer(tmpValueBuf, 32);
      pipe.InitBuffer(tmpIndexBuf, 32);
      pipe.InitBuffer(dstBuf, 32);
      auto localSrc = srcBuf.Get<float>();
      auto values = valueBuf.Get<float>();
      auto indices = indexBuf.Get<int32_t>();
      auto tmpValue = tmpValueBuf.Get<float>();
      auto tmpIndex = tmpIndexBuf.Get<int32_t>();
      auto dst = dstBuf.Get<int32_t>();
      uint32_t phase1Shape[] = {1, pitchR};
      // Replay Phase1 partitions in one simulator kernel, preserving their workspace layout.
      for (uint32_t n = 0; n < dimN; ++n) {
        GmToUb(localSrc, src + n * tileR, tileR);
        for (uint32_t r = tileR; r < pitchR; ++r) {
          localSrc.SetValue(r, 0.0F);
        }
        ArgMaxWithValueExtend<float, Pattern::Reduce::AR>(tmpIndex, tmpValue, localSrc, LocalTensor<uint8_t>(),
                                                          phase1Shape, false, n * tileR, tileR);
        ArgmaxCopy(values[n * phase2Pitch], tmpValue, 1);
        ArgmaxCopy(indices[n * phase2Pitch], tmpIndex, 1);
      }
      uint32_t phase2Shape[] = {dimN, phase2Pitch, phase2Pitch};
      ArgMaxMultiRPhase2Extend<float>(dst, values, indices, LocalTensor<uint8_t>(), phase2Shape, 1);
      UbToGm(index, dst, 1);
    };
    SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);
    EXPECT_EQ(index[0], scenario == 1 ? 1 : static_cast<int32_t>(dimR - 1)) << "scenario=" << scenario;
    EXPECT_GE(index[0], 0);
    EXPECT_LT(index[0], static_cast<int32_t>(dimR));
  }
  GmFree(src);
  GmFree(index);
}

// ArgMaxMultiRPhase2WithValueExtend 双输出（value+index）数值正确性：
// 覆盖 float/int32/int64 值类型，含并列、非对齐、多核，
// 以及 INT64_MIN/MAX 边界（int64 列特判数据）。
TEST(TestApiArgmax, Phase2WithValueDualOutput) {
  constexpr uint32_t dimN = 3;
  for (uint32_t columns : {1U, 8U, 65U, 136U}) {
    RunPhase2WithValue<float>(dimN, columns);
    RunPhase2WithValue<int32_t>(dimN, columns);
    RunPhase2WithValue<int64_t>(dimN, columns);
  }
}

// Phase2 双输入 dtype 不同导致的 value/index 行跨度不一致（value 8B vs index int32 4B）：
// 覆盖 A 非 8 的倍数（行跨度不同）与 A 为 8 倍数（行跨度相同）两种情况，value/index 双输出
// 与单输出变体都要正确按各自 dtype 行跨度寻址。
TEST(TestApiArgmax, Phase2DualPitch) {
  constexpr uint32_t dimN = 4;
  for (uint32_t aSize : {1U, 3U, 6U, 8U, 9U, 10U}) {
    RunPhase2DualPitch<float>(dimN, aSize);
    RunPhase2DualPitch<int32_t>(dimN, aSize);
    RunPhase2DualPitch<int64_t>(dimN, aSize);
  }
}

// ArgMaxMultiRPhase2Extend（单输出）沿核维度 N 按 value 选优、输出全局 index：
// 覆盖 float/int32 值类型（含并列取最小 index）。
TEST(TestApiArgmax, Phase2SingleOutput) {
  constexpr uint32_t dimN = 2;
  constexpr uint32_t dimA = 8;
  float *valueSrc = static_cast<float *>(GmAlloc(sizeof(float) * dimN * dimA));
  int32_t *indexSrc = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimN * dimA));
  int32_t *indexDst = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimA));
  std::fill_n(indexDst, dimA, static_cast<int32_t>(-1));
  // 核0 value=1；核1 value=2（胜）；indexSrc 为各核全局 index
  for (uint32_t a = 0; a < dimA; ++a) {
    valueSrc[0 * dimA + a] = 1.0F;
    valueSrc[1 * dimA + a] = 2.0F;
    indexSrc[0 * dimA + a] = static_cast<int32_t>(a);
    indexSrc[1 * dimA + a] = static_cast<int32_t>(dimA + a);
  }
  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> valueSrcBuf, indexSrcBuf, indexDstBuf;
    pipe.InitBuffer(valueSrcBuf, AlignBlock(sizeof(float) * dimN * dimA));
    pipe.InitBuffer(indexSrcBuf, AlignBlock(sizeof(int32_t) * dimN * dimA));
    pipe.InitBuffer(indexDstBuf, AlignBlock(sizeof(int32_t) * dimA));
    LocalTensor<float> valueSrcLocal = valueSrcBuf.Get<float>();
    LocalTensor<int32_t> indexSrcLocal = indexSrcBuf.Get<int32_t>();
    LocalTensor<int32_t> indexDstLocal = indexDstBuf.Get<int32_t>();
    GmToUb(valueSrcLocal, valueSrc, dimN * dimA);
    GmToUb(indexSrcLocal, indexSrc, dimN * dimA);
    // 本用例 value/index 按同一行跨度 dimA 排布，故两个跨度都传 dimA
    uint32_t srcShape[] = {dimN, dimA, dimA};
    ArgMaxMultiRPhase2Extend<float>(indexDstLocal, valueSrcLocal, indexSrcLocal, srcShape);
    UbToGm(indexDst, indexDstLocal, dimA);
  };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  for (uint32_t a = 0; a < dimA; ++a) {
    EXPECT_EQ(indexDst[a], static_cast<int32_t>(dimA + a)) << "a=" << a;
  }
  GmFree(valueSrc);
  GmFree(indexSrc);
  GmFree(indexDst);
}

// ArgmaxMergeRound 跨 tile 合并：覆盖值更新、index 平局，镜像 codegen 右值调用形态。
// 无 VL slack，覆盖同宽/混宽的完整 VL、尾块及小 A，验证合并与确定性语义。
TEST(TestApiArgmax, MergeRoundCrossTile) {
  for (uint32_t dimA : {1U, 3U, 6U, 31U, 32U, 33U, 63U, 64U, 65U, 100U}) {
    RunArgmaxMergeRound<float>(dimA);
    RunArgmaxMergeRound<int32_t>(dimA);
    RunArgmaxMergeRound<int64_t>(dimA);
  }
}

// ── Phase2 向量化（路线 2）测试 ─────────────────────────────────────────────
// 与 RunPhase2DualPitch 相同的数据布局（value/index 双行跨度 + 哨兵 padding），
// 但内核走 ArgMaxMultiRPhase2VectorExtend / ArgMaxMultiRPhase2WithValueVectorExtend
// （__simd_vf__：A 进 lane、标量走 N，完整 VL 连续加载，尾块 masked Gather）。
// 源 buffer 只按 32B 对齐分配，不能依赖额外 VL slack。

// Phase2 向量化（单输出）：value float/int32 × index int32
template <typename T>
void RunPhase2VectorSingle(uint32_t dimN, uint32_t aSize) {
  const uint32_t valueStep = 32U / sizeof(T);
  const uint32_t indexStep = 32U / sizeof(int32_t);
  const uint32_t dimAValue = (aSize + valueStep - 1U) / valueStep * valueStep;
  const uint32_t dimAIndex = (aSize + indexStep - 1U) / indexStep * indexStep;
  T *valueSrc = static_cast<T *>(GmAlloc(sizeof(T) * dimN * dimAValue));
  int32_t *indexSrc = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimN * dimAIndex));
  int32_t *indexDst = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimAIndex));
  std::fill_n(indexDst, dimAIndex, static_cast<int32_t>(-1));

  const T padValue = std::is_same_v<T, float> ? static_cast<T>(1.0e30F) : std::numeric_limits<T>::max();
  for (uint32_t n = 0; n < dimN; ++n) {
    for (uint32_t a = 0; a < dimAValue; ++a) {
      valueSrc[n * dimAValue + a] = padValue;
    }
    for (uint32_t a = 0; a < dimAIndex; ++a) {
      indexSrc[n * dimAIndex + a] = static_cast<int32_t>(-1000);
    }
  }
  for (uint32_t a = 0; a < aSize; ++a) {
    for (uint32_t n = 0; n < dimN; ++n) {
      valueSrc[n * dimAValue + a] = static_cast<T>(n + 1);
      indexSrc[n * dimAIndex + a] = static_cast<int32_t>(n * 1000 + a);
    }
    valueSrc[(a % dimN) * dimAValue + a] = static_cast<T>(dimN + 1);
  }

  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> vSrcBuf, iSrcBuf, iDstBuf;
    pipe.InitBuffer(vSrcBuf, AlignBlock(sizeof(T) * dimN * dimAValue));
    pipe.InitBuffer(iSrcBuf, AlignBlock(sizeof(int32_t) * dimN * dimAIndex));
    pipe.InitBuffer(iDstBuf, AlignBlock(sizeof(int32_t) * dimAIndex));
    LocalTensor<T> vSrcLocal = vSrcBuf.Get<T>();
    LocalTensor<int32_t> iSrcLocal = iSrcBuf.Get<int32_t>();
    LocalTensor<int32_t> iDstLocal = iDstBuf.Get<int32_t>();
    GmToUb(vSrcLocal, valueSrc, dimN * dimAValue);
    GmToUb(iSrcLocal, indexSrc, dimN * dimAIndex);
    uint32_t srcShape[] = {dimN, dimAValue, dimAIndex};
    ArgMaxMultiRPhase2VectorExtend<T>(iDstLocal, vSrcLocal, iSrcLocal, srcShape, aSize);
    UbToGm(indexDst, iDstLocal, dimAIndex);
  };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  for (uint32_t a = 0; a < aSize; ++a) {
    T best = valueSrc[a];
    int32_t bestIndex = indexSrc[a];
    for (uint32_t n = 1; n < dimN; ++n) {
      const T cur = valueSrc[n * dimAValue + a];
      const int32_t curIndex = indexSrc[n * dimAIndex + a];
      if (cur > best || (cur == best && curIndex < bestIndex)) {
        best = cur;
        bestIndex = curIndex;
      }
    }
    EXPECT_EQ(indexDst[a], bestIndex) << "single vec A=" << aSize << " a=" << a;
  }

  GmFree(valueSrc);
  GmFree(indexSrc);
  GmFree(indexDst);
}

// Phase2 向量化（WithValue 双输出）：value float/int32/int64 × index int32
template <typename T>
void RunPhase2VectorWithValue(uint32_t dimN, uint32_t aSize) {
  const uint32_t valueStep = 32U / sizeof(T);
  const uint32_t indexStep = 32U / sizeof(int32_t);
  const uint32_t dimAValue = (aSize + valueStep - 1U) / valueStep * valueStep;
  const uint32_t dimAIndex = (aSize + indexStep - 1U) / indexStep * indexStep;
  T *valueSrc = static_cast<T *>(GmAlloc(sizeof(T) * dimN * dimAValue));
  int32_t *indexSrc = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimN * dimAIndex));
  T *valueDst = static_cast<T *>(GmAlloc(sizeof(T) * dimAValue));
  int32_t *indexDst = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimAIndex));
  std::fill_n(valueDst, dimAValue, static_cast<T>(-1));
  std::fill_n(indexDst, dimAIndex, static_cast<int32_t>(-1));

  const T padValue = std::is_same_v<T, float> ? static_cast<T>(1.0e30F) : std::numeric_limits<T>::max();
  for (uint32_t n = 0; n < dimN; ++n) {
    for (uint32_t a = 0; a < dimAValue; ++a) {
      valueSrc[n * dimAValue + a] = padValue;
    }
    for (uint32_t a = 0; a < dimAIndex; ++a) {
      indexSrc[n * dimAIndex + a] = static_cast<int32_t>(-1000);
    }
  }
  for (uint32_t a = 0; a < aSize; ++a) {
    for (uint32_t n = 0; n < dimN; ++n) {
      valueSrc[n * dimAValue + a] = static_cast<T>(n + 1);
      indexSrc[n * dimAIndex + a] = static_cast<int32_t>(n * 1000 + a);
    }
    valueSrc[(a % dimN) * dimAValue + a] = static_cast<T>(dimN + 1);
  }

  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> vSrcBuf, iSrcBuf, vDstBuf, iDstBuf;
    pipe.InitBuffer(vSrcBuf, AlignBlock(sizeof(T) * dimN * dimAValue));
    pipe.InitBuffer(iSrcBuf, AlignBlock(sizeof(int32_t) * dimN * dimAIndex));
    pipe.InitBuffer(vDstBuf, AlignBlock(sizeof(T) * dimAValue));
    pipe.InitBuffer(iDstBuf, AlignBlock(sizeof(int32_t) * dimAIndex));
    LocalTensor<T> vSrcLocal = vSrcBuf.Get<T>();
    LocalTensor<int32_t> iSrcLocal = iSrcBuf.Get<int32_t>();
    LocalTensor<T> vDstLocal = vDstBuf.Get<T>();
    LocalTensor<int32_t> iDstLocal = iDstBuf.Get<int32_t>();
    GmToUb(vSrcLocal, valueSrc, dimN * dimAValue);
    GmToUb(iSrcLocal, indexSrc, dimN * dimAIndex);
    uint32_t srcShape[] = {dimN, dimAValue, dimAIndex};
    ArgMaxMultiRPhase2WithValueVectorExtend<T>(vDstLocal, iDstLocal, vSrcLocal, iSrcLocal, srcShape, aSize);
    UbToGm(valueDst, vDstLocal, dimAValue);
    UbToGm(indexDst, iDstLocal, dimAIndex);
  };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  for (uint32_t a = 0; a < aSize; ++a) {
    T best = valueSrc[a];
    int32_t bestIndex = indexSrc[a];
    for (uint32_t n = 1; n < dimN; ++n) {
      const T cur = valueSrc[n * dimAValue + a];
      const int32_t curIndex = indexSrc[n * dimAIndex + a];
      if (cur > best || (cur == best && curIndex < bestIndex)) {
        best = cur;
        bestIndex = curIndex;
      }
    }
    EXPECT_EQ(valueDst[a], best) << "withvalue vec A=" << aSize << " a=" << a;
    EXPECT_EQ(indexDst[a], bestIndex) << "withvalue vec A=" << aSize << " a=" << a;
  }

  GmFree(valueSrc);
  GmFree(indexSrc);
  GmFree(valueDst);
  GmFree(indexDst);
}

// 向量化 Phase2 单输出：
// - value 4B（float/int32）→ 同宽 __simd_vf__ 向量路径；A 覆盖 <vlSize、=vlSize 倍数、>vlSize 尾块。
// - value 8B（int64）→ 混宽（value 8B / index int32 4B）；块内 step=min(32,64)=32。
TEST(TestApiArgmax, Phase2VectorSingleOutput) {
  constexpr uint32_t dimN = 4;
  for (uint32_t aSize : {1U, 3U, 6U, 8U, 9U, 10U}) {
    RunPhase2VectorSingle<float>(dimN, aSize);
    RunPhase2VectorSingle<int32_t>(dimN, aSize);
  }
  // 混宽大 A：触发多 step 分块 + 尾块
  for (uint32_t aSize : {20U, 31U, 32U, 33U, 63U, 64U, 65U}) {
    RunPhase2VectorSingle<float>(dimN, aSize);
    RunPhase2VectorSingle<int32_t>(dimN, aSize);
  }
  // 多 vlSize 分块 + 尾块；65 已由上面的边界组合覆盖。
  RunPhase2VectorSingle<int32_t>(dimN, 136U);
}

// 向量化 Phase2 WithValue 双输出：
// - value 4B（float/int32）→ 同宽两个独立 vf 向量路径。
// - value 8B（int64）→ 混宽（value 8B / index int32 4B），step=min(32,64)=32。
TEST(TestApiArgmax, Phase2VectorWithValueDualOutput) {
  constexpr uint32_t dimN = 4;
  for (uint32_t aSize : {1U, 3U, 6U, 8U, 9U, 10U}) {
    RunPhase2VectorWithValue<float>(dimN, aSize);
    RunPhase2VectorWithValue<int32_t>(dimN, aSize);
    RunPhase2VectorWithValue<int64_t>(dimN, aSize);
  }
  // 混宽大 A：触发多 step 分块 + 尾块
  for (uint32_t aSize : {20U, 31U, 32U, 33U, 63U, 64U, 65U}) {
    RunPhase2VectorWithValue<float>(dimN, aSize);
    RunPhase2VectorWithValue<int32_t>(dimN, aSize);
    RunPhase2VectorWithValue<int64_t>(dimN, aSize);
  }
  // 多 vlSize 分块 + 尾块；65 已由上面的边界组合覆盖。
  RunPhase2VectorWithValue<int32_t>(dimN, 136U);
  RunPhase2VectorWithValue<int64_t>(dimN, 136U);
}

// Phase2 WithValue 混宽（value 8B / index 4B）+ A 非对齐：显式传 validA，
// 验证循环/mask/store 只处理真实 A 列，index/value 的 padding 区不被写。
template <typename T>
void RunPhase2MixedWidthValidA(uint32_t dimN, uint32_t dimA) {
  const uint32_t valueStep = 32U / sizeof(T);
  const uint32_t indexStep = 32U / sizeof(int32_t);
  const uint32_t dimAValue = (dimA + valueStep - 1U) / valueStep * valueStep;
  const uint32_t dimAIndex = (dimA + indexStep - 1U) / indexStep * indexStep;
  ASSERT_LT(dimAValue, dimAIndex);  // 本用例只针对 value 8B / index 4B 的 dimAIndex > dimAValue
  const T valuePad = std::numeric_limits<T>::max();
  const int32_t indexPad = static_cast<int32_t>(-1000);
  const T valueDstSentinel = static_cast<T>(-1);
  const int32_t indexDstSentinel = static_cast<int32_t>(-7);

  T *valueSrc = static_cast<T *>(GmAlloc(sizeof(T) * dimN * dimAValue));
  int32_t *indexSrc = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimN * dimAIndex));
  T *valueDst = static_cast<T *>(GmAlloc(sizeof(T) * dimAValue));
  int32_t *indexDst = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimAIndex));
  std::fill_n(valueDst, dimAValue, valueDstSentinel);
  std::fill_n(indexDst, dimAIndex, indexDstSentinel);

  for (uint32_t n = 0; n < dimN; ++n) {
    for (uint32_t a = 0; a < dimAValue; ++a) {
      valueSrc[n * dimAValue + a] = (a < dimA) ? static_cast<T>(a * 100 + n) : valuePad;
    }
    for (uint32_t a = 0; a < dimAIndex; ++a) {
      indexSrc[n * dimAIndex + a] = (a < dimA) ? static_cast<int32_t>(n * dimA + a) : indexPad;
    }
  }

  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> valueSrcBuf, indexSrcBuf, valueDstBuf, indexDstBuf;
    pipe.InitBuffer(valueSrcBuf, AlignBlock(sizeof(T) * dimN * dimAValue));
    pipe.InitBuffer(indexSrcBuf, AlignBlock(sizeof(int32_t) * dimN * dimAIndex));
    pipe.InitBuffer(valueDstBuf, AlignBlock(sizeof(T) * dimAValue));
    pipe.InitBuffer(indexDstBuf, AlignBlock(sizeof(int32_t) * dimAIndex));
    LocalTensor<T> valueSrcLocal = valueSrcBuf.Get<T>();
    LocalTensor<int32_t> indexSrcLocal = indexSrcBuf.Get<int32_t>();
    LocalTensor<T> valueDstLocal = valueDstBuf.Get<T>();
    LocalTensor<int32_t> indexDstLocal = indexDstBuf.Get<int32_t>();
    GmToUb(valueSrcLocal, valueSrc, dimN * dimAValue);
    GmToUb(indexSrcLocal, indexSrc, dimN * dimAIndex);
    GmToUb(valueDstLocal, valueDst, dimAValue);
    GmToUb(indexDstLocal, indexDst, dimAIndex);
    uint32_t srcShape[] = {dimN, dimAValue, dimAIndex};
    ArgMaxMultiRPhase2WithValueExtend<T>(valueDstLocal, indexDstLocal, valueSrcLocal, indexSrcLocal,
                                         LocalTensor<uint8_t>(), srcShape, dimA);
    UbToGm(valueDst, valueDstLocal, dimAValue);
    UbToGm(indexDst, indexDstLocal, dimAIndex);
  };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);

  const int32_t expectedIndex = static_cast<int32_t>((dimN - 1U) * dimA);
  for (uint32_t a = 0; a < dimA; ++a) {
    EXPECT_EQ(valueDst[a], static_cast<T>(a * 100 + (dimN - 1U))) << "A=" << dimA << " a=" << a;
    EXPECT_EQ(indexDst[a], static_cast<int32_t>(expectedIndex + a)) << "A=" << dimA << " a=" << a;
  }
  // validA 生效：padding 区不应被写（保持初值），证明没有越界 store。
  for (uint32_t a = dimA; a < dimAValue; ++a) {
    EXPECT_EQ(valueDst[a], valueDstSentinel) << "value padding written, A=" << dimA << " a=" << a;
  }
  for (uint32_t a = dimA; a < dimAIndex; ++a) {
    EXPECT_EQ(indexDst[a], indexDstSentinel) << "index padding written, A=" << dimA << " a=" << a;
  }

  GmFree(valueSrc);
  GmFree(indexSrc);
  GmFree(valueDst);
  GmFree(indexDst);
}

// value 8B（int64）/ index 4B（int32），A 非对齐 → dimAIndex > dimAValue；
// 回归此前按 dimAIndex 处理导致 value 跨行读、index padding 越界写的问题。
TEST(TestApiArgmax, Phase2WithValueMixedWidthValidA) {
  for (uint32_t dimN : {2U, 3U}) {
    for (uint32_t dimA : {1U, 3U, 9U, 33U}) {
      RunPhase2MixedWidthValidA<int64_t>(dimN, dimA);
    }
  }
}

}  // namespace af
