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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>
#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"
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
    const uint32_t scenario = row % kRows;
    for (uint32_t column = 0; column < columns; ++column) {
      T value = static_cast<T>(column);
      if (scenario == 1) {
        value = static_cast<T>(-3);
      } else if (scenario == 2) {
        const uint32_t second = columns > 1 ? columns - 2 : 0;
        value = column == std::min(5U, columns - 1) || column == second ? static_cast<T>(99) : static_cast<T>(-100);
      } else if (scenario == 3) {
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
        if (scenario == 4) {
          value = column % 2 == 0 ? -0.0F : 0.0F;
        } else if (scenario == 5) {
          value = column == 0 ? 0.0F : -0.0F;
        }
      }
      if constexpr (std::is_same_v<T, float>) {
        if (special) {
          if (scenario == 0) {
            value = std::numeric_limits<float>::quiet_NaN();
          } else if (scenario == 1) {
            value = column == 1 || column == columns - 1 ? std::numeric_limits<float>::quiet_NaN()
                                                         : std::numeric_limits<float>::infinity();
          } else if (scenario == 2) {
            value = -std::numeric_limits<float>::infinity();
          } else if (scenario == 3) {
            value = column == columns - 1 ? std::numeric_limits<float>::infinity() : 1.0F;
          } else if (scenario == 4) {
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
void RunArgmax(uint32_t columns, bool special = false, uint32_t validR = 0, uint32_t rows = kRows) {
  const uint32_t validColumns = validR == 0 ? columns : validR;
  SCOPED_TRACE(::testing::Message() << "A=" << rows << " R=" << columns << " validR=" << validR
                                    << " special=" << special);
  T *src = static_cast<T *>(GmAlloc(sizeof(T) * rows * columns));
  int32_t *index = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * (rows + 2)));
  std::fill_n(index, rows + 2, static_cast<int32_t>(-1));
  FillInput<T, isRa>(src, columns, special, rows);

  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuf, indexBuf;
    pipe.InitBuffer(srcBuf, AlignBlock(sizeof(T) * rows * columns));
    pipe.InitBuffer(indexBuf, AlignBlock(sizeof(int32_t) * (rows + 2)));
    LocalTensor<T> srcLocal = srcBuf.Get<T>();
    LocalTensor<int32_t> indexLocal = indexBuf.Get<int32_t>();
    GmToUb(srcLocal, src, rows * columns);
    // Nonzero high bits expose partial int64 stores; guard both ends of dst.
    GmToUb(indexLocal, index, rows + 2);
    uint32_t srcShape[] = {isRa ? columns : rows, isRa ? rows : columns};
    if constexpr (isRa) {
      ArgMaxExtend<T, Pattern::Reduce::RA>(indexLocal[1], srcLocal, srcShape, false, validR);
    } else {
      ArgMaxExtend<T, Pattern::Reduce::AR>(indexLocal[1], srcLocal, srcShape, false, validR);
    }
    UbToGm(index, indexLocal, rows + 2);
  };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  EXPECT_EQ(index[0], static_cast<int32_t>(-1));
  EXPECT_EQ(index[rows + 1], static_cast<int32_t>(-1));
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
    EXPECT_EQ(index[row + 1], static_cast<int32_t>(bestIndex)) << "row=" << row;
  }
  GmFree(src);
  GmFree(index);
}

// ArgMaxWithValueExtend（前端，1 入 2 出 value+index）：验证 value 为最大值、index 为首个最大位置。
// int64 value 走 ArgMaxWithValueExtend（static_assert 已放开 int64），非 ArgMaxExtend。
template <typename T, bool isRa>
void VerifyArgmaxWithValueRows(uint32_t rows, uint32_t columns, uint32_t validColumns, uint32_t indexBase, const T *src,
                               const T *value, const int32_t *index) {
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
    EXPECT_EQ(index[row + 1], static_cast<int32_t>(bestIndex + indexBase)) << "row=" << row;
  }
}

template <typename T, bool isRa>
void RunArgmaxWithValue(uint32_t columns, bool special = false, uint32_t validR = 0, uint32_t rows = kRows,
                        uint32_t indexBase = 0, bool minimum = false) {
  const uint32_t validColumns = validR == 0 ? columns : validR;
  SCOPED_TRACE(::testing::Message() << "A=" << rows << " R=" << columns << " validR=" << validR
                                    << " indexBase=" << indexBase << " special=" << special);
  T *src = static_cast<T *>(GmAlloc(sizeof(T) * rows * columns));
  int32_t *index = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * (rows + 2)));
  T *value = static_cast<T *>(GmAlloc(sizeof(T) * (rows + 2)));
  std::fill_n(index, rows + 2, static_cast<int32_t>(-1));
  std::fill_n(value, rows + 2, static_cast<T>(-1));
  FillInput<T, isRa>(src, columns, special, rows);
  if (minimum) {
    const T lowest = std::is_same_v<T, float> ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
    std::fill_n(src, rows * columns, lowest);
  }

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
      ArgMaxWithValueExtend<T, Pattern::Reduce::RA>(indexLocal[1], valueLocal[1], srcLocal, srcShape, false, indexBase,
                                                    validR);
    } else {
      ArgMaxWithValueExtend<T, Pattern::Reduce::AR>(indexLocal[1], valueLocal[1], srcLocal, srcShape, false, indexBase,
                                                    validR);
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
  VerifyArgmaxWithValueRows<T, isRa>(rows, columns, validColumns, indexBase, src, value, index);
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

template <typename T>
T MakeAccumulateSourceValue(uint32_t tile, uint32_t r, uint32_t a, uint32_t dimR,
                            const std::array<uint32_t, 3U> &valid) {
  T value = static_cast<T>(-100 + static_cast<int32_t>((r * 7U + a) % 31U));
  if (r >= valid[tile]) {
    return std::numeric_limits<T>::max();
  }
  if (a % 10U == 0U) {
    value = tile == a % 3U && r == valid[tile] - 1U ? T(100) : value;
  } else if (a % 10U == 1U) {
    value = r == 0U || r == valid[tile] - 1U ? T(100) : value;
  }
  if constexpr (std::is_same_v<T, float>) {
    if (r >= valid[tile] || (a % 10U == 2U && (r == 1U || r == valid[tile] - 1U)) ||
        (a % 10U == 3U && tile > 0U && r == valid[tile] - 1U)) {
      uint32_t bits = 0x7fc00001U + tile * dimR + r;
      std::copy_n(reinterpret_cast<const char *>(&bits), sizeof(value), reinterpret_cast<char *>(&value));
    } else if (a % 10U == 4U) {
      value = tile == 0U ? -0.0F : 0.0F;
    } else if (a % 10U == 5U) {
      value = tile == 0U ? 0.0F : -0.0F;
    } else if (a % 10U == 6U) {
      value = -std::numeric_limits<float>::infinity();
    } else if (a % 10U == 7U) {
      value = std::numeric_limits<float>::infinity();
    }
  }
  return value;
}

template <typename T, bool isRa>
void FillAccumulateSource(uint32_t dimA, uint32_t dimR, const std::array<uint32_t, 3U> &valid, std::vector<T> &source) {
  for (uint32_t tile = 0; tile < 3U; ++tile) {
    for (uint32_t a = 0; a < dimA; ++a) {
      for (uint32_t r = 0; r < dimR; ++r) {
        source[tile * dimR * dimA + (isRa ? r * dimA + a : a * dimR + r)] =
            MakeAccumulateSourceValue<T>(tile, r, a, dimR, valid);
      }
    }
  }
}

template <typename T, bool isRa>
void LaunchAccumulateKernel(uint32_t dimA, uint32_t dimR, uint32_t guard, uint32_t outputSize, bool reverseBases,
                            const std::array<uint32_t, 3U> &valid, T *input, T *observedInput, T *values,
                            int32_t *indices, T *temporaryValues, int32_t *temporaryIndices) {
  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuf, valueBuf, indexBuf, tmpValueBuf, tmpIndexBuf;
    pipe.InitBuffer(srcBuf, AlignBlock(sizeof(T) * dimR * dimA));
    pipe.InitBuffer(valueBuf, AlignBlock(sizeof(T) * outputSize));
    pipe.InitBuffer(indexBuf, AlignBlock(sizeof(int32_t) * outputSize));
    pipe.InitBuffer(tmpValueBuf, AlignBlock(sizeof(T) * outputSize));
    pipe.InitBuffer(tmpIndexBuf, AlignBlock(sizeof(int32_t) * outputSize));
    auto src = srcBuf.Get<T>();
    auto value = valueBuf.Get<T>();
    auto index = indexBuf.Get<int32_t>();
    auto tmpValue = tmpValueBuf.Get<T>();
    auto tmpIndex = tmpIndexBuf.Get<int32_t>();
    GmToUb(value, values, outputSize);
    GmToUb(index, indices, outputSize);
    GmToUb(tmpValue, values, outputSize);
    GmToUb(tmpIndex, indices, outputSize);
    const uint32_t shape[] = {isRa ? dimR : dimA, isRa ? dimA : dimR};
    using PatternT = std::conditional_t<isRa, Pattern::Reduce::RA, Pattern::Reduce::AR>;
    for (uint32_t tile = 0; tile < 3U; ++tile) {
      GmToUb(src, input + tile * dimR * dimA, dimR * dimA);
      const uint32_t base = 73U + (reverseBases ? 2U - tile : tile) * dimR;
      ArgMaxWithValueAccumulateExtend<T, PatternT>(index[guard], value[guard], src, tmpIndex[guard], tmpValue[guard],
                                                   shape, base, valid[tile], tile == 0U);
      UbToGm(observedInput + tile * dimR * dimA, src, dimR * dimA);
      UbToGm(values + tile * outputSize, value, outputSize);
      UbToGm(indices + tile * outputSize, index, outputSize);
    }
    UbToGm(temporaryValues, tmpValue, outputSize);
    UbToGm(temporaryIndices, tmpIndex, outputSize);
  };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
}

template <typename T, bool isRa>
void VerifyAccumulateRow(uint32_t round, uint32_t a, uint32_t dimA, uint32_t dimR, uint32_t guard, uint32_t outputSize,
                         bool reverseBases, const std::array<uint32_t, 3U> &valid, const std::vector<T> &source,
                         const T *values, const int32_t *indices) {
  auto at = [&](uint32_t tile, uint32_t r, uint32_t col) -> const T & {
    return source[tile * dimR * dimA + (isRa ? r * dimA + col : col * dimR + r)];
  };
  T best = at(0, 0, a);
  uint32_t bestIndex = 73U + (reverseBases ? 2U : 0U) * dimR;
  for (uint32_t tile = 0; tile <= round; ++tile) {
    const uint32_t base = 73U + (reverseBases ? 2U - tile : tile) * dimR;
    for (uint32_t r = 0; r < valid[tile]; ++r) {
      const T candidate = at(tile, r, a);
      bool equal = candidate == best;
      if constexpr (std::is_same_v<T, float>) {
        equal = equal || (std::isnan(candidate) && std::isnan(best));
      }
      if (IsLarger(candidate, best) || (equal && base + r < bestIndex)) {
        best = candidate;
        bestIndex = base + r;
      }
    }
  }
  const uint32_t offset = round * outputSize + guard + a;
  EXPECT_EQ(indices[offset], static_cast<int32_t>(bestIndex)) << "round=" << round << " a=" << a;
  EXPECT_EQ(std::memcmp(&values[offset], &best, sizeof(T)), 0) << "round=" << round << " a=" << a;
}

template <typename T, bool isRa>
void VerifyAccumulateTiles(uint32_t dimA, uint32_t dimR, uint32_t guard, uint32_t outputSize, bool reverseBases,
                           const std::array<uint32_t, 3U> &valid, const std::vector<T> &source, const T *values,
                           const int32_t *indices, const T *temporaryValues, const int32_t *temporaryIndices) {
  if (isRa && dimA % (GetVecLen() / sizeof(T)) == 0U) {
    for (uint32_t i = 0; i < outputSize; ++i) {
      EXPECT_EQ(temporaryValues[i], T(999));
      EXPECT_EQ(temporaryIndices[i], -999);
    }
  }
  for (uint32_t round = 0; round < 3U; ++round) {
    for (uint32_t a = 0; a < dimA; ++a) {
      VerifyAccumulateRow<T, isRa>(round, a, dimA, dimR, guard, outputSize, reverseBases, valid, source, values,
                                   indices);
    }
    for (uint32_t i = 0; i < guard; ++i) {
      for (uint32_t offset : {round * outputSize + i, round * outputSize + guard + dimA + i}) {
        EXPECT_EQ(values[offset], T(999));
        EXPECT_EQ(indices[offset], -999);
      }
    }
  }
}

template <typename T, bool isRa>
void RunAccumulateTiles(uint32_t dimA, uint32_t dimR, bool reverseBases = false) {
  SCOPED_TRACE(::testing::Message() << "RA=" << isRa << " A=" << dimA << " R=" << dimR << " reverse=" << reverseBases);
  constexpr uint32_t rounds = 3U;
  const std::array<uint32_t, 3U> valid = {dimR, dimR > 1U ? dimR - 1U : 1U, std::min(3U, dimR)};
  const uint32_t guard = GetVecLen() / sizeof(T);
  const uint32_t outputSize = dimA + 2U * guard;
  std::vector<T> source(rounds * dimR * dimA);
  FillAccumulateSource<T, isRa>(dimA, dimR, valid, source);
  T *input = static_cast<T *>(GmAlloc(sizeof(T) * source.size()));
  std::copy(source.begin(), source.end(), input);
  T *observedInput = static_cast<T *>(GmAlloc(sizeof(T) * source.size()));
  T *values = static_cast<T *>(GmAlloc(sizeof(T) * outputSize * rounds));
  int32_t *indices = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * outputSize * rounds));
  T *temporaryValues = static_cast<T *>(GmAlloc(sizeof(T) * outputSize));
  int32_t *temporaryIndices = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * outputSize));
  std::fill_n(values, outputSize * rounds, T(999));
  std::fill_n(indices, outputSize * rounds, -999);
  LaunchAccumulateKernel<T, isRa>(dimA, dimR, guard, outputSize, reverseBases, valid, input, observedInput, values,
                                  indices, temporaryValues, temporaryIndices);
  EXPECT_EQ(std::memcmp(input, source.data(), sizeof(T) * source.size()), 0);
  EXPECT_EQ(std::memcmp(observedInput, source.data(), sizeof(T) * source.size()), 0);
  VerifyAccumulateTiles<T, isRa>(dimA, dimR, guard, outputSize, reverseBases, valid, source, values, indices,
                                 temporaryValues, temporaryIndices);
  GmFree(input);
  GmFree(observedInput);
  GmFree(values);
  GmFree(indices);
  GmFree(temporaryValues);
  GmFree(temporaryIndices);
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

TEST(TestApiArgmax, RAVectorKeepFirstWinner) {
  // Full-R ARA tiles after layout conversion: ties, signed zeros and first/later NaNs.
  // Include single-row R and a masked physical tail, plus nonzero global index bases.
  for (uint32_t rows : {16U, 64U, 65U}) {
    for (uint32_t columns : {1U, 2U, 32U}) {
      RunArgmaxWithValue<float, true>(columns, true, 0, rows, 97);
      RunArgmaxWithValue<int32_t, true>(columns, true, 0, rows, 97);
      RunArgmaxWithValue<int64_t, true>(columns, true, 0, rows, 97);
    }
    RunArgmaxWithValue<float, true>(33, true, 32, rows, 97);
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

TEST(TestApiArgmax, SinglePassRowBoundaryMatrix) {
  // Exercise the shared AR scanner at both sides of the aligned-load boundary.
  for (uint32_t columns : {1U, 31U, 32U, 33U, 63U, 64U, 65U, 127U, 128U, 129U, 2048U}) {
    RunArgmaxWithValue<float, false>(columns, false, 0, 1);
  }
  for (uint32_t columns : {33U, 65U, 129U, 2048U}) {
    RunArgmaxWithValue<int64_t, false>(columns, false, 0, 1);
  }
  // Physical R is aligned, but only three lanes are valid; poisoned tail lanes must not win.
  RunArgmaxWithValue<float, false>(8U, true, 3U, 1);

  // Narrow RA rows use the shared gather scanner, including exact VL and partial-VL cases.
  for (uint32_t columns : {1U, 2U, 3U, 6U, 7U, 31U, 32U, 33U, 63U, 64U, 65U, 129U}) {
    RunArgmaxWithValue<float, true>(columns, false, 0, 16);
  }
}

TEST(TestApiArgmax, SinglePassMasksInactiveLanesNearIndexLimit) {
  constexpr uint32_t base = static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) - 3U;
  RunArgmaxWithValue<float, false>(8U, false, 3U, kRows, base, true);
  RunArgmaxWithValue<int32_t, false>(8U, false, 3U, kRows, base, true);
  RunArgmaxWithValue<int64_t, false>(8U, false, 3U, kRows, base, true);
}

TEST(TestApiArgmax, RAVectorIndexOnly) {
  // Full A vectors select the RA branch; repeat scenarios in every vector block.
  for (uint32_t rows : {64U, 128U}) {
    for (uint32_t columns : {1U, 3U, 65U}) {
      RunArgmax<float, true>(columns, false, 0, rows);
      RunArgmax<int32_t, true>(columns, false, 0, rows);
      RunArgmax<float, true>(columns, true, 0, rows);
      // Physical padding includes a larger numeric value and a trailing NaN.
      RunArgmax<float, true>(columns + 1, false, columns, rows);
      RunArgmax<int32_t, true>(columns + 1, false, columns, rows);
      RunArgmax<float, true>(columns + 1, true, columns, rows);
    }
  }
}

TEST(TestApiArgmax, RAVectorWithValue) {
  for (uint32_t rows : {64U, 128U}) {
    for (uint32_t columns : {1U, 3U, 65U}) {
      RunArgmaxWithValue<float, true>(columns, false, 0, rows);
      RunArgmaxWithValue<int32_t, true>(columns, false, 0, rows);
      RunArgmaxWithValue<float, true>(columns, true, 0, rows);
      RunArgmaxWithValue<float, true>(columns + 1, false, columns, rows, 97);
      RunArgmaxWithValue<int32_t, true>(columns + 1, false, columns, rows, 97);
      RunArgmaxWithValue<float, true>(columns + 1, true, columns, rows, 97);
    }
  }
}

TEST(TestApiArgmax, RAVectorWithValueInt64) {
  // int64 values use 32 lanes; indices remain int32 with guarded output stores.
  for (uint32_t rows : {32U, 64U}) {
    for (uint32_t columns : {1U, 3U, 65U}) {
      RunArgmaxWithValue<int64_t, true>(columns, false, 0, rows);
      RunArgmaxWithValue<int64_t, true>(columns + 1, false, columns, rows, 97);
    }
  }
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

// Phase2 向量化（单输出）：value float/int32/int64 × index int32
template <typename T>
void FillPhase2VectorSingleSource(uint32_t dimN, uint32_t aSize, uint32_t dimAValue, uint32_t dimAIndex, T *valueSrc,
                                  int32_t *indexSrc) {
  if (dimN != 0U) {
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
      if (a % 3U == 0U && dimN > 1U) {
        valueSrc[a] = valueSrc[(dimN - 1U) * dimAValue + a] = static_cast<T>(dimN + 1);
        indexSrc[a] = static_cast<int32_t>(10000U + a);
        indexSrc[(dimN - 1U) * dimAIndex + a] = static_cast<int32_t>(a);
      }
    }
  } else {
    FAIL() << "dimN must be positive.";
  }
}

TEST(TestApiArgmax, Phase2VectorSourceRejectsZeroRowsBeforeAccess) {
  EXPECT_FATAL_FAILURE(FillPhase2VectorSingleSource<float>(0U, 1U, 1U, 1U, nullptr, nullptr), "dimN");
  EXPECT_FATAL_FAILURE(FillPhase2VectorSingleSource<int64_t>(0U, 1U, 1U, 1U, nullptr, nullptr), "dimN");
}

template <typename T>
void VerifyPhase2VectorSingle(uint32_t dimN, uint32_t aSize, uint32_t dimAValue, uint32_t dimAIndex, const T *valueSrc,
                              const int32_t *indexSrc, const int32_t *indexDst) {
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
  for (uint32_t a = aSize; a < dimAIndex; ++a) {
    EXPECT_EQ(indexDst[a], -1) << "single vec padding A=" << aSize << " a=" << a;
  }
}

template <typename T>
void RunPhase2VectorSingle(uint32_t dimN, uint32_t aSize) {
  ASSERT_GT(dimN, 0U);
  const uint32_t valueStep = 32U / sizeof(T);
  const uint32_t indexStep = 32U / sizeof(int32_t);
  const uint32_t dimAValue = (aSize + valueStep - 1U) / valueStep * valueStep;
  const uint32_t dimAIndex = (aSize + indexStep - 1U) / indexStep * indexStep;
  T *valueSrc = static_cast<T *>(GmAlloc(sizeof(T) * dimN * dimAValue));
  int32_t *indexSrc = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimN * dimAIndex));
  int32_t *indexDst = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * dimAIndex));
  std::fill_n(indexDst, dimAIndex, static_cast<int32_t>(-1));

  FillPhase2VectorSingleSource<T>(dimN, aSize, dimAValue, dimAIndex, valueSrc, indexSrc);

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
    GmToUb(iDstLocal, indexDst, dimAIndex);
    uint32_t srcShape[] = {dimN, dimAValue, dimAIndex};
    ArgMaxMultiRPhase2VectorExtend<T>(iDstLocal, vSrcLocal, iSrcLocal, srcShape, aSize);
    UbToGm(indexDst, iDstLocal, dimAIndex);
  };

  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  VerifyPhase2VectorSingle<T>(dimN, aSize, dimAValue, dimAIndex, valueSrc, indexSrc, indexDst);

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
    RunPhase2VectorSingle<int64_t>(dimN, aSize);
  }
  // 混宽大 A：触发多 step 分块 + 尾块
  for (uint32_t aSize : {20U, 31U, 32U, 33U, 63U, 64U, 65U}) {
    RunPhase2VectorSingle<float>(dimN, aSize);
    RunPhase2VectorSingle<int32_t>(dimN, aSize);
    RunPhase2VectorSingle<int64_t>(dimN, aSize);
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

TEST(TestApiArgmax, AccumulateRaFullVectors) {
  for (uint32_t dimA : {64U, 128U, 512U}) {
    for (uint32_t dimR : {1U, 3U, 4U, 7U, 55U}) {
      RunAccumulateTiles<float, true>(dimA, dimR);
    }
  }
}

TEST(TestApiArgmax, AccumulateRaEarlierGlobalIndices) {
  RunAccumulateTiles<float, true>(128U, 7U, true);
}

TEST(TestApiArgmax, AccumulateLayoutFallbacks) {
  for (uint32_t dimA : {1U, 13U, 65U}) {
    RunAccumulateTiles<float, true>(dimA, 7U);
    RunAccumulateTiles<float, true>(dimA, 7U, true);
    RunAccumulateTiles<float, false>(dimA, 7U);
    RunAccumulateTiles<float, false>(dimA, 8U);
  }
}

TEST(TestApiArgmax, AccumulateIntegerWidths) {
  RunAccumulateTiles<int32_t, true>(128U, 7U);
  RunAccumulateTiles<int64_t, true>(64U, 7U);
  RunAccumulateTiles<int64_t, true>(13U, 7U, true);
  RunAccumulateTiles<int64_t, false>(13U, 7U);
}

template <typename T>
T MakePitchedSourceValue(uint32_t r, uint32_t a, uint32_t rows, uint32_t scenario) {
  T v = static_cast<T>(r + a);
  if constexpr (std::is_same_v<T, int64_t>) {
    v += (static_cast<int64_t>(1) << 40);
  }
  if (scenario == 1U) {
    v = r == std::min(3U, rows - 1U) || r == rows - 1U ? static_cast<T>(99) : static_cast<T>(-3);
  } else if (scenario == 2U) {
    v = std::numeric_limits<T>::lowest();
  } else if (scenario == 3U) {
    v = r == std::min(2U, rows - 1U) ? std::numeric_limits<T>::max() : static_cast<T>(-4);
    if constexpr (std::is_same_v<T, float>) {
      if (r == std::min(2U, rows - 1U) || r == rows - 1U) {
        const uint32_t bits = 0x7FC01234U;
        std::copy_n(reinterpret_cast<const char *>(&bits), sizeof(v), reinterpret_cast<char *>(&v));
      }
    }
  } else if (scenario == 4U) {
    v = r % 2U == 0U ? static_cast<T>(-0.0F) : static_cast<T>(0.0F);
  } else if (scenario == 5U) {
    v = static_cast<T>(-static_cast<int64_t>(r));
    if constexpr (std::is_same_v<T, float>) {
      v = -std::numeric_limits<float>::infinity();
    }
  }
  return v;
}

template <typename T>
void FillPitchedSource(T *source, uint32_t pitch, uint32_t logicalA, uint32_t rows, uint32_t scenario) {
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t a = 0; a < logicalA; ++a) {
      source[r * pitch + a] = MakePitchedSourceValue<T>(r, a, rows, scenario);
    }
  }
}

template <typename T>
void RunPitchedArgmaxApi(int api, const LocalTensor<int32_t> &indexLocal, const LocalTensor<T> &valueLocal,
                         const LocalTensor<T> &sourceLocal, const LocalTensor<int32_t> &temporaryIndexLocal,
                         const LocalTensor<T> &temporaryValueLocal, const uint32_t *shape, uint32_t firstBase,
                         uint32_t secondBase, uint32_t rows, uint32_t logicalA) {
  if (api == 0) {
    if constexpr (!std::is_same_v<T, int64_t>) {
      ArgMaxExtend<T, Pattern::Reduce::RA>(indexLocal[1], sourceLocal, shape, false, rows, logicalA);
    }
  } else if (api == 1) {
    ArgMaxWithValueExtend<T, Pattern::Reduce::RA>(indexLocal[1], valueLocal[1], sourceLocal, shape, false, firstBase,
                                                  rows, logicalA);
  } else if (api == 2) {
    ArgMaxWithValueExtend<T, Pattern::Reduce::RA>(indexLocal[1], valueLocal[1], sourceLocal, LocalTensor<uint8_t>(),
                                                  shape, false, firstBase, rows, logicalA);
  } else {
    ArgMaxWithValueAccumulateExtend<T, Pattern::Reduce::RA>(indexLocal[1], valueLocal[1], sourceLocal,
                                                            temporaryIndexLocal[1], temporaryValueLocal[1], shape,
                                                            firstBase, rows, true, logicalA);
  }
}

template <typename T>
void LaunchPitchedKernel(int api, T *source, T *value, T *temporaryValue, T *firstValue, int32_t *index,
                         int32_t *temporaryIndex, int32_t *firstIndex, uint32_t pitch, uint32_t logicalA, uint32_t rows,
                         uint32_t sourceCount, uint32_t sourceBytes, uint32_t valueSlots, uint32_t indexSlots,
                         uint32_t firstBase, uint32_t secondBase) {
  auto kernel = [=] {
    TPipe pipe;
    TBuf<TPosition::VECCALC> sourceBuf, indexBuf, valueBuf, temporaryIndexBuf, temporaryValueBuf;
    pipe.InitBuffer(sourceBuf, sourceBytes);
    pipe.InitBuffer(indexBuf, indexSlots * sizeof(int32_t));
    pipe.InitBuffer(valueBuf, valueSlots * sizeof(T));
    pipe.InitBuffer(temporaryIndexBuf, indexSlots * sizeof(int32_t));
    pipe.InitBuffer(temporaryValueBuf, valueSlots * sizeof(T));
    // Place the final logical element at the end of the allocation: no final-row or VL slack.
    auto sourceLocal = sourceBuf.Get<T>()[sourceBytes / sizeof(T) - sourceCount];
    auto indexLocal = indexBuf.Get<int32_t>();
    auto valueLocal = valueBuf.Get<T>();
    auto temporaryIndexLocal = temporaryIndexBuf.Get<int32_t>();
    auto temporaryValueLocal = temporaryValueBuf.Get<T>();
    GmToUb(sourceLocal, source, sourceCount);
    GmToUb(indexLocal, index, indexSlots);
    GmToUb(valueLocal, value, valueSlots);
    GmToUb(temporaryIndexLocal, temporaryIndex, indexSlots);
    GmToUb(temporaryValueLocal, temporaryValue, valueSlots);
    uint32_t shape[] = {rows, pitch};
    RunPitchedArgmaxApi<T>(api, indexLocal, valueLocal, sourceLocal, temporaryIndexLocal, temporaryValueLocal, shape,
                           firstBase, secondBase, rows, logicalA);
    if (api == 3) {
      UbToGm(firstIndex, indexLocal, indexSlots);
      UbToGm(firstValue, valueLocal, valueSlots);
      ArgMaxWithValueAccumulateExtend<T, Pattern::Reduce::RA>(indexLocal[1], valueLocal[1], sourceLocal,
                                                              temporaryIndexLocal[1], temporaryValueLocal[1], shape,
                                                              secondBase, rows, false, logicalA);
    }
    UbToGm(index, indexLocal, indexSlots);
    UbToGm(value, valueLocal, valueSlots);
    UbToGm(temporaryIndex, temporaryIndexLocal, indexSlots);
    UbToGm(temporaryValue, temporaryValueLocal, valueSlots);
  };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
}

template <typename T>
void VerifyPitchedResults(const T *source, const T *value, const T *temporaryValue, const T *firstValue,
                          const int32_t *index, const int32_t *temporaryIndex, const int32_t *firstIndex,
                          uint32_t pitch, uint32_t logicalA, uint32_t rows, int api, uint32_t firstBase,
                          uint32_t secondBase, uint32_t valueSlots, uint32_t indexSlots, T sentinel) {
  constexpr int32_t indexSentinel = -91;
  for (uint32_t a = 0; a < logicalA; ++a) {
    uint32_t winner = 0U;
    for (uint32_t r = 1; r < rows; ++r) {
      if (IsLarger(source[r * pitch + a], source[winner * pitch + a])) {
        winner = r;
      }
    }
    const T expected = source[winner * pitch + a];
    const uint32_t base = api == 0 ? 0U : (api == 3 ? std::min(firstBase, secondBase) : firstBase);
    EXPECT_EQ(index[a + 1U], static_cast<int32_t>(base + winner));
    if (api != 0) {
      EXPECT_EQ(std::memcmp(&value[a + 1U], &expected, sizeof(T)), 0);
    }
    if (api == 3) {
      EXPECT_EQ(firstIndex[a + 1U], static_cast<int32_t>(firstBase + winner));
      EXPECT_EQ(std::memcmp(&firstValue[a + 1U], &expected, sizeof(T)), 0);
      EXPECT_EQ(temporaryIndex[a + 1U], static_cast<int32_t>(secondBase + winner));
      EXPECT_EQ(std::memcmp(&temporaryValue[a + 1U], &expected, sizeof(T)), 0);
    }
  }
  // Every allocated output/tmp padding element is a checked sentinel, not unused oversized output.
  for (uint32_t i = 0; i < indexSlots; ++i) {
    if (i == 0U || i > logicalA) {
      EXPECT_EQ(index[i], indexSentinel);
      EXPECT_EQ(temporaryIndex[i], indexSentinel);
    }
  }
  for (uint32_t i = 0; i < valueSlots; ++i) {
    if (i == 0U || i > logicalA) {
      EXPECT_EQ(value[i], sentinel);
      EXPECT_EQ(temporaryValue[i], sentinel);
    }
  }
}

template <typename T>
void RunPitchedArgmaxValidA(uint32_t pitch, uint32_t logicalA, uint32_t rows, uint32_t scenario, int api,
                            bool reverseBase) {
  SCOPED_TRACE(::testing::Message() << "pitch=" << pitch << " validA=" << logicalA << " R=" << rows
                                    << " scenario=" << scenario << " api=" << api << " reverse=" << reverseBase);
  const uint32_t sourceCount = (rows - 1U) * pitch + logicalA;
  const uint32_t sourceBytes = AlignBlock(sourceCount * sizeof(T));
  const uint32_t valueSlots = AlignBlock((logicalA + 2U) * sizeof(T)) / sizeof(T);
  const uint32_t indexSlots = AlignBlock((logicalA + 2U) * sizeof(int32_t)) / sizeof(int32_t);
  const T sentinel = static_cast<T>(-73);
  constexpr int32_t indexSentinel = -91;
  T *source = static_cast<T *>(GmAlloc(sourceCount * sizeof(T)));
  T *value = static_cast<T *>(GmAlloc(valueSlots * sizeof(T)));
  T *temporaryValue = static_cast<T *>(GmAlloc(valueSlots * sizeof(T)));
  T *firstValue = static_cast<T *>(GmAlloc(valueSlots * sizeof(T)));
  int32_t *index = static_cast<int32_t *>(GmAlloc(indexSlots * sizeof(int32_t)));
  int32_t *temporaryIndex = static_cast<int32_t *>(GmAlloc(indexSlots * sizeof(int32_t)));
  int32_t *firstIndex = static_cast<int32_t *>(GmAlloc(indexSlots * sizeof(int32_t)));
  std::fill_n(source, sourceCount, std::numeric_limits<T>::max());
  FillPitchedSource<T>(source, pitch, logicalA, rows, scenario);
  std::fill_n(value, valueSlots, sentinel);
  std::fill_n(temporaryValue, valueSlots, sentinel);
  std::fill_n(index, indexSlots, indexSentinel);
  std::fill_n(temporaryIndex, indexSlots, indexSentinel);
  const uint32_t firstBase = reverseBase ? 137U : 7U;
  const uint32_t secondBase = reverseBase ? 7U : 137U;
  LaunchPitchedKernel<T>(api, source, value, temporaryValue, firstValue, index, temporaryIndex, firstIndex, pitch,
                         logicalA, rows, sourceCount, sourceBytes, valueSlots, indexSlots, firstBase, secondBase);
  VerifyPitchedResults<T>(source, value, temporaryValue, firstValue, index, temporaryIndex, firstIndex, pitch, logicalA,
                          rows, api, firstBase, secondBase, valueSlots, indexSlots, sentinel);
  GmFree(source);
  GmFree(value);
  GmFree(temporaryValue);
  GmFree(firstValue);
  GmFree(index);
  GmFree(temporaryIndex);
  GmFree(firstIndex);
}

TEST(TestApiArgmax, PitchedValidAExactSourceFloat) {
  for (const uint32_t pitch : {4U, 8U, 16U, 64U, 128U}) {
    for (const uint32_t rows : {1U, 7U, 65U, 129U}) {
      for (uint32_t scenario = 0; scenario < 6U; ++scenario) {
        for (int api = 0; api < 4; ++api) {
          RunPitchedArgmaxValidA<float>(pitch, 1U, rows, scenario, api, false);
          if (api == 3) {
            RunPitchedArgmaxValidA<float>(pitch, 1U, rows, scenario, api, true);
          }
        }
      }
    }
  }
}

TEST(TestApiArgmax, PitchedValidAExactSourceIntegerWidths) {
  for (const uint32_t pitch : {4U, 8U, 16U, 64U}) {
    for (const uint32_t logicalA : {1U, 3U}) {
      for (uint32_t scenario = 0; scenario < 6U; ++scenario) {
        for (int api = 0; api < 4; ++api) {
          RunPitchedArgmaxValidA<int32_t>(pitch, logicalA, 65U, scenario, api, true);
          if (api != 0) {
            RunPitchedArgmaxValidA<int64_t>(pitch, logicalA, 65U, scenario, api, true);
          }
        }
      }
    }
  }
}

}  // namespace af
