/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "tikicpulib.h"
#include "kernel_operator.h"
using namespace AscendC;

#include "utils.h"
#include "test_api_utils.h"
#include "argmax.h"

class TestApiArgmax : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiArgmax, Test_argmax_float_ar) {
  // 构造测试输入和预期结果
  int a = this->GetParam()[0];
  int b = this->GetParam()[1];
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a);
  int64_t expect[a];

  for (int i = 0; i < a; i++) {
    float maxVal = -1e30f;
    int64_t maxIdx = 0;
    for (int j = 0; j < b; j++) {
      x[i * b + j] = (float)(i * b + j);
      if (x[i * b + j] > maxVal) {
        maxVal = x[i * b + j];
        maxIdx = j;
      }
    }
    expect[i] = maxIdx;
  }

  // 构造Api调用函数
  auto kernel = [](int32_t a, int32_t b, float *x, int64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(float) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * a);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<float>();
    auto l_y = ybuf.Get<int64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a);

    uint32_t shape[] = {static_cast<uint32_t>(a), static_cast<uint32_t>(b)};

    ArgMaxExtend<int64_t, float, AscendC::Pattern::Reduce::AR>(l_y, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, a);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST_P(TestApiArgmax, Test_argmax_int32_ar) {
  // 构造测试输入和预期结果
  int a = this->GetParam()[0];
  int b = this->GetParam()[1];
  auto *x = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a);
  int64_t expect[a];

  for (int i = 0; i < a; i++) {
    int32_t maxVal = -2147483647;
    int64_t maxIdx = 0;
    for (int j = 0; j < b; j++) {
      x[i * b + j] = (int32_t)(i * b + j);
      if (x[i * b + j] > maxVal) {
        maxVal = x[i * b + j];
        maxIdx = j;
      }
    }
    expect[i] = maxIdx;
  }

  // 构造Api调用函数
  auto kernel = [](int32_t a, int32_t b, int32_t *x, int64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int32_t) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * a);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<int32_t>();
    auto l_y = ybuf.Get<int64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a);

    uint32_t shape[] = {static_cast<uint32_t>(a), static_cast<uint32_t>(b)};

    ArgMaxExtend<int64_t, int32_t, AscendC::Pattern::Reduce::AR>(l_y, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, a);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, TestApiArgmax,
                         ::testing::Values(std::vector<int>{16, 32}, std::vector<int>{32, 64}, std::vector<int>{8, 128},
                                           std::vector<int>{4, 72}));

TEST(TestApiArgmaxRA, Test_argmax_float_ra) {
  // 构造测试输入和预期结果
  uint32_t a = 16, b = 64;
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * b);
  int64_t expect[b];

  for (uint32_t i = 0; i < a * b; i++) {
    x[i] = (float)i;
  }

  for (uint32_t j = 0; j < b; j++) {
    float maxVal = -1e30f;
    int64_t maxIdx = 0;
    for (uint32_t i = 0; i < a; i++) {
      uint32_t idx = i * b + j;
      if (x[idx] > maxVal) {
        maxVal = x[idx];
        maxIdx = i;
      }
    }
    expect[j] = maxIdx;
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t a, uint32_t b, float *x, int64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(float) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * b);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<float>();
    auto l_y = ybuf.Get<int64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, b);

    uint32_t shape[] = {a, b};

    ArgMaxExtend<int64_t, float, AscendC::Pattern::Reduce::RA>(l_y, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < b; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiArgmaxRA, Test_argmax_int32_ra) {
  // 构造测试输入和预期结果
  uint32_t a = 16, b = 64;
  auto *x = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * b);
  int64_t expect[b];

  for (uint32_t i = 0; i < a * b; i++) {
    x[i] = (int32_t)i;
  }

  for (uint32_t j = 0; j < b; j++) {
    int32_t maxVal = -2147483647;
    int64_t maxIdx = 0;
    for (uint32_t i = 0; i < a; i++) {
      uint32_t idx = i * b + j;
      if (x[idx] > maxVal) {
        maxVal = x[idx];
        maxIdx = i;
      }
    }
    expect[j] = maxIdx;
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t a, uint32_t b, int32_t *x, int64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int32_t) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * b);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<int32_t>();
    auto l_y = ybuf.Get<int64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, b);

    uint32_t shape[] = {a, b};

    ArgMaxExtend<int64_t, int32_t, AscendC::Pattern::Reduce::RA>(l_y, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < b; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiArgmaxLE64, Test_argmax_float_le64) {
  // 构造测试输入和预期结果 - 测试last维度小于等于64的情况
  uint32_t a = 32, b = 32;
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a);
  int64_t expect[a];

  for (uint32_t i = 0; i < a; i++) {
    float maxVal = -1e30f;
    int64_t maxIdx = 0;
    for (uint32_t j = 0; j < b; j++) {
      x[i * b + j] = (float)(i * b + j);
      if (x[i * b + j] > maxVal) {
        maxVal = x[i * b + j];
        maxIdx = j;
      }
    }
    expect[i] = maxIdx;
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t a, uint32_t b, float *x, int64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(float) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * a);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<float>();
    auto l_y = ybuf.Get<int64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a);

    uint32_t shape[] = {a, b};

    ArgMaxExtend<int64_t, float, AscendC::Pattern::Reduce::AR>(l_y, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, a);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < a; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiArgmaxLE64, Test_argmax_int32_le64) {
  // 构造测试输入和预期结果 - 测试last维度小于等于64的情况
  uint32_t a = 32, b = 32;
  auto *x = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a);
  int64_t expect[a];

  for (uint32_t i = 0; i < a; i++) {
    int32_t maxVal = -2147483647;
    int64_t maxIdx = 0;
    for (uint32_t j = 0; j < b; j++) {
      x[i * b + j] = (int32_t)(i * b + j);
      if (x[i * b + j] > maxVal) {
        maxVal = x[i * b + j];
        maxIdx = j;
      }
    }
    expect[i] = maxIdx;
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t a, uint32_t b, int32_t *x, int64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int32_t) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * a);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<int32_t>();
    auto l_y = ybuf.Get<int64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a);

    uint32_t shape[] = {a, b};

    ArgMaxExtend<int64_t, int32_t, AscendC::Pattern::Reduce::AR>(l_y, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, a);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < a; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}
