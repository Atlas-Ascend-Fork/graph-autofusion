/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_isfinite.cpp
 * \brief
 */

#include <cmath>
#include "gtest/gtest.h"
#include "kernel_operator.h"
#include "tikicpulib.h"

#include "test_api_utils.h"
#include "utils.h"
#include "cast.h"

using namespace AscendC;

constexpr int gen_index_two = 2;
constexpr int gen_index_three = 3;
constexpr int gen_index_five = 5;
constexpr int gen_index_div = 1000;
constexpr float gen_float_suffix = 0.12;

TEST(TestApiCast, LargeDstRepeatStrideFallsBackToLoop) {
  constexpr uint32_t kFirstDim = 2;
  constexpr uint32_t kLastDim = 8;
  constexpr uint32_t kInputLastDimStride = 16;
  constexpr uint32_t kOutputLastDimStride = 2048;
  constexpr uint32_t kInputSize = kFirstDim * kInputLastDimStride;
  constexpr uint32_t kOutputSize = kFirstDim * kOutputLastDimStride;
  constexpr float kSentinel = -1.0F;
  auto *x = static_cast<half *>(AscendC::GmAlloc(sizeof(half) * kInputSize));
  auto *y = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kOutputSize));
  for (uint32_t i = 0; i < kInputSize; ++i) {
    x[i] = 0;
  }
  for (uint32_t i = 0; i < kLastDim; ++i) {
    x[i] = 1;
    x[kInputLastDimStride + i] = 2;
  }
  for (uint32_t i = 0; i < kOutputSize; ++i) {
    y[i] = kSentinel;
  }

  auto kernel = [](half *x, float *y) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(half) * kInputSize);
    tpipe.InitBuffer(ybuf, sizeof(float) * kOutputSize);
    tpipe.InitBuffer(tmp, 8192);
    auto l_x = xbuf.Get<half>();
    auto l_y = ybuf.Get<float>();
    auto l_tmp = tmp.Get<uint8_t>();
    GmToUb(l_x, x, kInputSize);
    GmToUb(l_y, y, kOutputSize);
    CastExtend(l_y, l_x, l_tmp, kFirstDim, kLastDim, kInputLastDimStride, kOutputLastDimStride, sizeof(float));
    UbToGm(y, l_y, kOutputSize);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, x, y);

  for (uint32_t i = 0; i < kLastDim; ++i) {
    EXPECT_EQ(y[i], 1.0F);
    EXPECT_EQ(y[kOutputLastDimStride + i], 2.0F);
  }
  EXPECT_EQ(y[kLastDim], kSentinel);
  EXPECT_EQ(y[kOutputLastDimStride + kLastDim], kSentinel);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

template <typename InT, typename OutT>
void CastExtendCalc(InT *x, OutT *y, int size) {
  TPipe tpipe;
  TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
  tpipe.InitBuffer(xbuf, sizeof(InT) * size);
  tpipe.InitBuffer(ybuf, sizeof(OutT) * size);
  tpipe.InitBuffer(tmp, 8192);

  auto l_x = xbuf.Get<InT>();
  auto l_y = ybuf.Get<OutT>();
  auto l_tmp = tmp.Get<uint8_t>();

  GmToUb(l_x, x, size);
  GmToUb(l_y, y, size);

  CastExtend(l_y, l_x, l_tmp, size);

  UbToGm(y, l_y, size);
}

template <typename InT, typename OutT>
void CastExtendTest(int size, std::function<OutT(int index, InT src)> expectGen, std::function<InT(int index)> srcGen,
                    std::function<bool(OutT a, OutT b)> compare) {
  // 构造测试输入和预期结果
  auto *x = (InT *)AscendC::GmAlloc(sizeof(InT) * size);
  auto *y = (OutT *)AscendC::GmAlloc(sizeof(OutT) * size);

  OutT expect[size];

  for (int i = 0; i < size; i++) {
    auto srcGenValue = srcGen(i);
    auto expectGenValue = expectGen(i, x[i]);
    x[i] = srcGenValue;
    expect[i] = expectGenValue;
  }

  // 构造Api调用函数
  auto kernel = [](int size, InT *x, OutT *y) { CastExtendCalc<InT, OutT>(x, y, size); };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, size, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < size; i++) {
    auto xValue = x[i];
    auto yValue = y[i];
    auto expectValue = expect[i];
    if (!compare(yValue, expectValue)) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}

template <typename InT, typename OutT>
void CastExtendWithMaskModeCalc(InT *x, OutT *y, int first_dim, int last_dim, int input_last_dim_stride,
                                int output_last_dim_stride) {
  TPipe tpipe;
  TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
  tpipe.InitBuffer(xbuf, sizeof(InT) * first_dim * input_last_dim_stride);
  tpipe.InitBuffer(ybuf, sizeof(OutT) * first_dim * output_last_dim_stride);
  tpipe.InitBuffer(tmp, 8192);

  auto l_x = xbuf.Get<InT>();
  auto l_y = ybuf.Get<OutT>();
  auto l_tmp = tmp.Get<uint8_t>();

  GmToUb(l_x, x, first_dim * input_last_dim_stride);
  CastExtend(l_y, l_x, l_tmp, first_dim, last_dim, input_last_dim_stride, output_last_dim_stride,
             std::max(sizeof(InT), sizeof(OutT)));
  UbToGm(y, l_y, first_dim * output_last_dim_stride);
}

TEST(TestApiCast, Int64ToHalfWithPaddedLastDim) {
  constexpr int first_dim = 2;
  constexpr int last_dim = 3;
  constexpr int input_last_dim_stride = ONE_BLK_SIZE / sizeof(int64_t);
  constexpr int output_last_dim_stride = ONE_BLK_SIZE / sizeof(half);
  auto *x = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * first_dim * input_last_dim_stride));
  auto *y = static_cast<half *>(AscendC::GmAlloc(sizeof(half) * first_dim * output_last_dim_stride));

  for (int row = 0; row < first_dim; ++row) {
    for (int column = 0; column < last_dim; ++column) {
      x[row * input_last_dim_stride + column] = row * last_dim + column;
    }
  }

  auto kernel = [](int64_t *input, half *output) {
    CastExtendWithMaskModeCalc<int64_t, half>(input, output, first_dim, last_dim, input_last_dim_stride,
                                              output_last_dim_stride);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, x, y);

  for (int row = 0; row < first_dim; ++row) {
    for (int column = 0; column < last_dim; ++column) {
      EXPECT_EQ(static_cast<int64_t>(static_cast<float>(y[row * output_last_dim_stride + column])),
                row * last_dim + column);
    }
  }
}

constexpr uint8_t CastGen(const int index) {
  return index % gen_index_two;
}

constexpr bool CastCompare(const float a, const float b) {
  auto diff = a - b;
  if (diff < -1e-5 || diff > 1e-5) {
    return false;
  }
  return true;
}

constexpr float ExpectGenU82F32(const int index, const float x) {
  return (float)(index % gen_index_two);
}

class TestApiCastUint8 : public ::testing::Test, public testing::WithParamInterface<size_t> {};

TEST_P(TestApiCastUint8, Calc) {
  const int size = this->GetParam();
  CastExtendTest<uint8_t, float>(size, ExpectGenU82F32, CastGen, CastCompare);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiCastUint8,
                         ::testing::Values(
                             /* 1 block */ ONE_BLK_SIZE / sizeof(float),
                             /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(float),
                             /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(float),
                             /* less than 1 block */ (ONE_BLK_SIZE - sizeof(float)) / sizeof(float),
                             /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float),
                             /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float),
                             /* mix block, repeat, max repeat*/
                             ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(float))) /
                                 sizeof(float)));

constexpr int64_t CastGenS642U8(const int index) {
  return index % gen_index_div;
}

constexpr bool CastCompareS642U8(const uint8_t a, const uint8_t b) {
  return (a - b == 0);
}

constexpr uint8_t ExpectGenS642U8(const int index, const uint8_t x) {
  auto tmp = index % gen_index_div;
  if (tmp > 255) {
    return 255;
  } else {
    return tmp;
  }
}

class TestApiCastS642U8 : public ::testing::Test, public testing::WithParamInterface<size_t> {};

TEST_P(TestApiCastS642U8, Calc) {
  const int size = this->GetParam();
  CastExtendTest<int64_t, uint8_t>(size, ExpectGenS642U8, CastGenS642U8, CastCompareS642U8);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiCastS642U8,
                         ::testing::Values(
                             /* 1 block */ ONE_BLK_SIZE / sizeof(int64_t),
                             /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* less than 1 block */ (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                             /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t),
                             /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* mix block, repeat, max repeat*/
                             ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(int64_t))) /
                                 sizeof(int64_t)));

constexpr int64_t CastGenS642F16(const int index) {
  return index % gen_index_div;
}

constexpr bool CastCompareS642F16(const double a, const double b) {
  auto diff = a - b;
  if (diff < -1e-5 || diff > 1e-5) {
    return false;
  }
  return true;
}

constexpr double ExpectGenS642F16(const int index, const uint8_t x) {
  return double(index % gen_index_div);
}

class TestApiCastS642F16 : public ::testing::Test, public testing::WithParamInterface<size_t> {};

TEST_P(TestApiCastS642F16, Calc) {
  const int size = this->GetParam();
  CastExtendTest<int64_t, half>(size, ExpectGenS642F16, CastGenS642F16, CastCompareS642F16);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiCastS642F16,
                         ::testing::Values(
                             /* 1 block */ ONE_BLK_SIZE / sizeof(int64_t),
                             /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* less than 1 block */ (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                             /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t),
                             /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* mix block, repeat, max repeat*/
                             ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(int64_t))) /
                                 sizeof(int64_t)));

constexpr float CastGenF162S64(const int index) {
  if (index % gen_index_two == 0) {
    return index / gen_index_div + gen_float_suffix;
  }
  if (index % gen_index_three == 0) {
    return (index / gen_index_div + gen_float_suffix) * -1;
  }
  return index / gen_index_div;
}

constexpr bool CastCompareF162S64(const int64_t a, const int64_t b) {
  return (a == b);
}

constexpr int64_t ExpectGenF162S64(const int index, const float x) {
  if (index % gen_index_two == 0) {
    return int64_t(index / gen_index_div + gen_float_suffix);
  }
  if (index % gen_index_three == 0) {
    return int64_t((index / gen_index_div + gen_float_suffix) * -1);
  }
  return int64_t(index / gen_index_div);
}

class TestApiCastF162S64 : public ::testing::Test, public testing::WithParamInterface<size_t> {};

TEST_P(TestApiCastF162S64, Calc) {
  const int size = this->GetParam();
  CastExtendTest<half, int64_t>(size, ExpectGenF162S64, CastGenF162S64, CastCompareF162S64);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiCastF162S64,
                         ::testing::Values(
                             /* 1 block */ ONE_BLK_SIZE / sizeof(int64_t),
                             /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* less than 1 block */ (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                             /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t),
                             /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* mix block, repeat, max repeat*/
                             ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(int64_t))) /
                                 sizeof(int64_t)));

constexpr uint32_t CastGenUint2Int(const int index) {
  if (index % gen_index_two == 0) {
    return (uint32_t)0xFFFFFF85;
  }
  if (index % gen_index_three == 0) {
    return uint32_t((index / gen_index_div) * 100 - 1);
  }
  return uint32_t(index / gen_index_div);
}

constexpr bool CastCompareUint2Int(const int32_t a, const int32_t b) {
  return (a == b);
}

constexpr int32_t ExpectGenUint2Int(const int index, const uint32_t x) {
  if (index % gen_index_two == 0) {
    return int32_t(0xFFFFFF85);
  }
  if (index % gen_index_three == 0) {
    return int32_t((index / gen_index_div) * 100 - 1);
  }
  return int32_t(index / gen_index_div);
}

class TestApiCastUint2Int : public ::testing::Test, public testing::WithParamInterface<size_t> {};

TEST_P(TestApiCastUint2Int, Calc) {
  const int size = this->GetParam();
  CastExtendTest<uint32_t, int32_t>(size, ExpectGenUint2Int, CastGenUint2Int, CastCompareUint2Int);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiCastUint2Int,
                         ::testing::Values(
                             /* 1 block */ ONE_BLK_SIZE / sizeof(int32_t),
                             /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(int32_t),
                             /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(int32_t),
                             /* less than 1 block */ (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
                             /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t),
                             /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t),
                             /* mix block, repeat, max repeat*/
                             ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(int32_t))) /
                                 sizeof(int32_t)));

constexpr int64_t CastGenS642U64(const int index) {
  if (index % gen_index_two == 0) {
    return (int64_t)0xFFFFFF85;
  }
  if (index % gen_index_three == 0) {
    return int64_t((index / gen_index_div) * 100 - 1);
  }
  return int64_t(index / gen_index_div);
}

constexpr bool CastCompareS642U64(const int32_t a, const int32_t b) {
  return (a == b);
}

constexpr uint64_t ExpectGenS642U64(const int index, const int64_t x) {
  if (index % gen_index_two == 0) {
    return uint64_t(0xFFFFFF85);
  }
  if (index % gen_index_three == 0) {
    return uint64_t((index / gen_index_div) * 100 - 1);
  }
  return uint64_t(index / gen_index_div);
}

class TestApiCastS642U64 : public ::testing::Test, public testing::WithParamInterface<size_t> {};

TEST_P(TestApiCastS642U64, Calc) {
  const int size = this->GetParam();
  CastExtendTest<int64_t, uint64_t>(size, ExpectGenS642U64, CastGenS642U64, CastCompareS642U64);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiCastS642U64,
                         ::testing::Values(
                             /* 1 block */ ONE_BLK_SIZE / sizeof(int64_t),
                             /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* less than 1 block */ (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                             /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t),
                             /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* mix block, repeat, max repeat*/
                             ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(int64_t))) /
                                 sizeof(int64_t)));

constexpr uint8_t CastGenU82S64(const int index) {
  return static_cast<uint8_t>(index);
}

constexpr bool CastCompareU82S64(const int64_t a, const int64_t b) {
  return (a == b);
}

constexpr int64_t ExpectGenU82S64(const int index, const uint8_t x) {
  return static_cast<int64_t>(static_cast<uint8_t>(index));
}

class TestApiCastU82S64 : public ::testing::Test, public testing::WithParamInterface<size_t> {};

TEST_P(TestApiCastU82S64, Calc) {
  const int size = this->GetParam();
  CastExtendTest<uint8_t, int64_t>(size, ExpectGenU82S64, CastGenU82S64, CastCompareU82S64);
}

// size 以 dst(int64) 为单位：CastExtendTest 会申请 sizeof(int64_t) * size 的输出 UB，若按 src(uint8)
// 口径取 max repeat(65280) 会使输出 buffer 溢出 UB。以 int64 为口径时最大 size(8160) 仍大于
// CastWithTwoTransfer 的不重叠 chunk（考虑中间缓冲 32B 对齐后为 1360），可覆盖多轮 calc_loop + 尾块。
INSTANTIATE_TEST_SUITE_P(DiffLength, TestApiCastU82S64,
                         ::testing::Values(
                             /* 1 block */ ONE_BLK_SIZE / sizeof(int64_t),
                             /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* max repeat */ MAX_REPEAT_NUM *ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* less than 1 block */ (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t),
                             /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t),
                             /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t),
                             /* mix block, repeat, max repeat*/
                             ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                              (ONE_BLK_SIZE - sizeof(int64_t))) /
                                 sizeof(int64_t)));
