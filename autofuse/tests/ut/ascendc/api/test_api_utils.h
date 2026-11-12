/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TEST_API_UTILS_H__
#define __TEST_API_UTILS_H__

#include <gtest/gtest.h>
#include "kernel_operator.h"
using namespace AscendC;
#include "utils.h"

template <class T>
void GmToUb(LocalTensor<T> &local, T *gm, int size) {
  for (int i = 0; i < size; i++) {
    local.SetValue(i, gm[i]);
  }
}

template <class T>
void UbToGm(T *gm, LocalTensor<T> &local, int size) {
  for (int i = 0; i < size; i++) {
    gm[i] = local.GetValue(i);
  }
}

template <class T>
void GmToUbGeneral(LocalTensor<T> &dst, T *src, int a, int b, int c, int d, int padD) {
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        int inputStart = ((i * b + j) * c + k) * d;
        int outputStart = ((i * b + j) * c + k) * padD;
        for (int l = 0; l < d; l++) {
          dst.SetValue(outputStart + l, src[inputStart + l]);
        }
      }
    }
  }
}

template <class T>
void UbToGmGeneral(T *dst, LocalTensor<T> &src, int a, int b, int c, int d, int padD) {
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        int inputStart = ((i * b + j) * c + k) * padD;
        int outputStart = ((i * b + j) * c + k) * d;
        for (int l = 0; l < d; l++) {
          dst[outputStart + l] = src.GetValue(inputStart + l);
        }
      }
    }
  }
}

template <typename T>
void UnaryCalc(
    T *x, T *y, int size,
    std::function<void(const LocalTensor<T> &x, const LocalTensor<T> &y, LocalTensor<uint8_t> &tmp, uint32_t size)>
        calc) {
  TPipe tpipe;
  TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
  tpipe.InitBuffer(xbuf, sizeof(T) * size);
  tpipe.InitBuffer(ybuf, sizeof(T) * size);
  tpipe.InitBuffer(tmp, TMP_UB_SIZE);

  auto l_x = xbuf.Get<T>();
  auto l_y = ybuf.Get<T>();
  auto l_tmp = tmp.Get<uint8_t>();

  GmToUb(l_x, x, size);
  GmToUb(l_y, y, size);

  calc(l_y, l_x, l_tmp, size);

  UbToGm(y, l_y, size);
}

constexpr inline double DefaultSrcGen(int index) {
  return index + 1;
}

constexpr inline double DefaultCompare(double a, double b) {
  const double atol = 1e-5;
  return (a - b) < atol && (a - b) > (-atol);
}

template <typename T>
void UnaryTest(
    int size,
    std::function<void(const LocalTensor<T> &x, const LocalTensor<T> &y, LocalTensor<uint8_t> &tmp, uint32_t size)>
        calc,
    std::function<double(double src)> expectGen, std::function<double(int index)> srcGen = DefaultSrcGen,
    std::function<double(double a, double b)> compare = DefaultCompare) {
  // 构造测试输入和预期结果
  auto *x = (T *)AscendC::GmAlloc(sizeof(T) * size);
  auto *y = (T *)AscendC::GmAlloc(sizeof(T) * size);

  T expect[size];

  for (int i = 0; i < size; i++) {
    x[i] = srcGen(i);
    expect[i] = expectGen(x[i]);
  }

  // 构造Api调用函数
  auto kernel = [&calc](int size, T *x, T *y) { UnaryCalc<T>(x, y, size, calc); };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, size, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < size; i++) {
    if (!compare(y[i], expect[i])) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

template <typename InT, typename OutT>
void UnaryCalc(
    InT *x, OutT *y, int size,
    std::function<void(const LocalTensor<OutT> &y, const LocalTensor<InT> &x, LocalTensor<uint8_t> &tmp, uint32_t size)>
        calc) {
  TPipe tpipe;
  TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
  tpipe.InitBuffer(xbuf, sizeof(InT) * size);
  tpipe.InitBuffer(ybuf, sizeof(OutT) * size);
  tpipe.InitBuffer(tmp, TMP_UB_SIZE);

  auto l_x = xbuf.Get<InT>();
  auto l_y = ybuf.Get<OutT>();
  auto l_tmp = tmp.Get<uint8_t>();

  GmToUb(l_x, x, size);
  GmToUb(l_y, y, size);

  calc(l_y, l_x, l_tmp, size);

  UbToGm(y, l_y, size);
}

template <typename InT, typename OutT>
void UnaryTest(
    int size,
    std::function<void(const LocalTensor<OutT> &y, const LocalTensor<InT> &x, LocalTensor<uint8_t> &tmp, uint32_t size)>
        calc,
    std::function<OutT(int index, InT src)> expectGen, std::function<InT(int index)> srcGen = DefaultSrcGen,
    std::function<bool(OutT a, OutT b)> compare = DefaultCompare) {
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
  auto kernel = [&calc](int size, InT *x, OutT *y) { UnaryCalc<InT, OutT>(x, y, size, calc); };

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
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

template <typename T, typename T_ONE>
void UnaryCalc(T *x, T *y, int size,
               std::function<void(LocalTensor<T> &x, LocalTensor<T> &y, LocalTensor<T_ONE> &ones,
                                  LocalTensor<uint8_t> &tmp, int size)>
                   calc) {
  TPipe tpipe;
  TBuf<TPosition::VECCALC> xbuf, ybuf, tmp, ones;
  tpipe.InitBuffer(xbuf, sizeof(T) * size);
  tpipe.InitBuffer(ybuf, sizeof(T) * size);
  tpipe.InitBuffer(tmp, TMP_UB_SIZE);
  tpipe.InitBuffer(ones, DEFAULT_C0_SIZE);

  auto l_x = xbuf.Get<T>();
  auto l_y = ybuf.Get<T>();
  auto l_tmp = tmp.Get<uint8_t>();
  auto l_one = ones.Get<uint8_t>();
  auto one_buf = l_one.template ReinterpretCast<T_ONE>();
  Duplicate(one_buf[0], (T_ONE)1.0, ONE_BLK_SIZE / sizeof(T_ONE));

  GmToUb(l_x, x, size);
  GmToUb(l_y, y, size);

  calc(l_y, l_x, one_buf, l_tmp, size);

  UbToGm(y, l_y, size);
}

template <typename T, typename T_ONE>
void UnaryTest(int size,
               std::function<void(LocalTensor<T> &x, LocalTensor<T> &y, LocalTensor<T_ONE> &ones,
                                  LocalTensor<uint8_t> &tmp, int size)>
                   calc,
               std::function<double(double src)> expectGen, std::function<double(int index)> srcGen = DefaultSrcGen,
               std::function<double(double a, double b)> compare = DefaultCompare) {
  // 构造测试输入和预期结果
  auto *x = (T *)AscendC::GmAlloc(sizeof(T) * size);
  auto *y = (T *)AscendC::GmAlloc(sizeof(T) * size);

  T expect[size];

  for (int i = 0; i < size; i++) {
    x[i] = srcGen(i);
    expect[i] = expectGen(x[i]);
  }

  // 构造Api调用函数
  auto kernel = [&calc](int size, T *x, T *y) { UnaryCalc<T, T_ONE>(x, y, size, calc); };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, size, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < size; i++) {
    if (!compare(y[i], expect[i])) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

// 版本4: 单模板参数，类型一致的 expectGen/srcGen/compare，calc 参数顺序为 (y, x, tmp, size)
template <typename T>
void UnaryTest(
    int size,
    std::function<void(const LocalTensor<T> &y, const LocalTensor<T> &x, LocalTensor<uint8_t> &tmp, uint32_t size)>
        calc,
    std::function<T(int index, T src)> expectGen, std::function<T(int index)> srcGen,
    std::function<bool(T a, T b)> compare) {
  auto *x = (T *)AscendC::GmAlloc(sizeof(T) * size);
  auto *y = (T *)AscendC::GmAlloc(sizeof(T) * size);

  T expect[size];

  for (int i = 0; i < size; i++) {
    auto srcGenValue = srcGen(i);
    auto expectGenValue = expectGen(i, srcGenValue);
    x[i] = srcGenValue;
    expect[i] = expectGenValue;
  }

  auto kernel = [&calc](int size, T *x, T *y) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(T) * size);
    tpipe.InitBuffer(ybuf, sizeof(T) * size);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    auto l_x = xbuf.Get<T>();
    auto l_y = ybuf.Get<T>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, size);
    GmToUb(l_y, y, size);

    calc(l_y, l_x, l_tmp, size);

    UbToGm(y, l_y, size);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, size, x, y);

  int diff_count = 0;
  for (int i = 0; i < size; i++) {
    if (!compare(y[i], expect[i])) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}
#endif
