/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * test_where.cpp
 */

#include <cmath>
#include "gtest/gtest.h"
#include "test_api_utils.h"
#include "tikicpulib.h"
#include "utils.h" #noqa
#include "duplicate.h" #noqa
#include "where.h"

using namespace AscendC;

namespace af {
template <typename T>
struct WhereInputParam {
  T *y{};
  uint8_t *x1{};
  T *x2{};
  T *x3{};
  T *exp{};
  uint32_t size{};
  uint32_t x2_size{};
  uint32_t x3_size{};
  // normal
  uint32_t m{};
  uint32_t y_stride{};
  uint32_t x1_stride{};
  uint32_t x2_stride{};
  uint32_t x3_stride{};
  bool x2_bcast{};
  bool x3_bcast{};
  float *x2_{};
  float *x3_{};
};

class TestApiWhereUT : public testing::Test {
 protected:
  template <typename T>
  static void CreateInput(WhereInputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = (T *)AscendC::GmAlloc(sizeof(T) * param.size);
    param.x1 = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * param.size);
    param.x2 = (T *)AscendC::GmAlloc(sizeof(T) * param.x2_size);
    param.x3 = (T *)AscendC::GmAlloc(sizeof(T) * param.x3_size);
    param.exp = (T *)AscendC::GmAlloc(sizeof(T) * param.size);
    int mask_value_range = 2;
    int input_value_range = 20000;
    int input_offset = 2;
    for (int i = 0; i < param.size; i++) {
      param.x1[i] = static_cast<uint8_t>(i % mask_value_range);
      if (i == 0 || param.x2_size > 1) {
        param.x2[i] = static_cast<T>((i + input_offset) % input_value_range);
        if constexpr (std::is_same_v<T, int64_t>) {
          param.x2[i] += 20000001;
        }
      }
      if (i == 0 || param.x3_size > 1) {
        param.x3[i] = static_cast<T>(-((i + input_offset) % input_value_range));
        if constexpr (std::is_same_v<T, int64_t>) {
          param.x3[i] -= 20000001;
        }
      }
      uint32_t x2_index = param.x2_size > 1 ? i : 0;
      uint32_t x3_index = param.x3_size > 1 ? i : 0;
      param.exp[i] = param.x1[i] == 1 ? param.x2[x2_index] : param.x3[x3_index];
    }
  }

  template <typename T>
  static uint32_t Valid(WhereInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (!DefaultCompare(param.y[i], param.exp[i])) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void InvokeKernel(WhereInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, x3buf, ybuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(uint8_t) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(T) * param.x2_size);
    tpipe.InitBuffer(x3buf, sizeof(T) * param.x3_size);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<uint8_t> l_x1 = x1buf.Get<uint8_t>();
    LocalTensor<T> l_x2 = x2buf.Get<T>();
    LocalTensor<T> l_x3 = x3buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb<uint8_t>(l_x1, param.x1, param.size);
    GmToUb(l_x2, param.x2, param.x2_size);
    GmToUb(l_x3, param.x3, param.x3_size);
    GmToUb(l_y, param.y, param.size);

    if (param.x2_size == 1 && param.x3_size == 1) {
      Where(l_y, l_x1, (T)param.x2[0], (T)param.x3[0], param.size, l_tmp);
    } else if (param.x2_size == 1) {
      Where(l_y, l_x1, (T)param.x2[0], l_x3, param.size, l_tmp);
    } else if (param.x3_size == 1) {
      Where(l_y, l_x1, l_x2, (T)param.x3[0], param.size, l_tmp);
    } else {
      Where(l_y, l_x1, l_x2, l_x3, param.size, l_tmp);
    }

    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void WhereTest(uint32_t size, uint32_t x2_size = 0, uint32_t x3_size = 0) {
    WhereInputParam<T> param{};
    param.size = size;
    param.x2_size = x2_size == 0 ? size : x2_size;
    param.x3_size = x3_size == 0 ? size : x3_size;

    CreateInput(param);

    // 构造Api调用函数
    auto kernel = [&param]() { InvokeKernel<T>(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.x1);
    AscendC::GmFree(param.x2);
    AscendC::GmFree(param.x3);
    AscendC::GmFree(param.exp);
  }
  // normal
  template <typename T>
  static void CreateNormalInput(WhereInputParam<T> &param) {
    // 构造测试输入和预期结果
    uint32_t y_align = (param.size * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    uint32_t x1_align = (param.size * sizeof(uint8_t) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    uint32_t x2_align =
        param.x2_bcast ? ONE_BLK_SIZE : (param.x2_size * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    uint32_t x3_align =
        param.x3_bcast ? ONE_BLK_SIZE : (param.x3_size * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;

    param.y_stride = y_align / sizeof(T);
    param.x1_stride = x1_align / sizeof(uint8_t);
    param.x2_stride = param.x2_bcast ? ONE_BLK_SIZE / sizeof(float) : x2_align / sizeof(T);
    param.x3_stride = param.x3_bcast ? ONE_BLK_SIZE / sizeof(float) : x3_align / sizeof(T);

    param.y = (T *)AscendC::GmAlloc(y_align * param.m);
    param.x1 = (uint8_t *)AscendC::GmAlloc(x1_align * param.m);
    param.x2 = (T *)AscendC::GmAlloc(x2_align * param.m);
    param.x3 = (T *)AscendC::GmAlloc(x3_align * param.m);
    param.exp = (T *)AscendC::GmAlloc(y_align * param.m);
    param.x2_ = (float *)AscendC::GmAlloc(x2_align);
    param.x3_ = (float *)AscendC::GmAlloc(x3_align);
    int mask_value_range = 2;
    int input_value_range = 20000;
    int input_offset = 2;
    for (int k = 0; k < param.m; k++) {
      for (int i = 0; i < param.size; i++) {
        param.x1[k * param.x1_stride + i] = static_cast<uint8_t>(i % mask_value_range);
        if (!param.x2_bcast) {
          param.x2[k * param.x2_stride + i] = static_cast<T>((i + input_offset) % input_value_range);
        } else if (i < param.x2_size) {
          param.x2_[i] = static_cast<float>((0 + input_offset) % input_value_range);
        }
        if (!param.x3_bcast) {
          param.x3[k * param.x3_stride + i] = static_cast<T>(-((i + input_offset) % input_value_range));
        } else if (i < param.x3_size) {
          param.x3_[i] = static_cast<float>(-((0 + input_offset) % input_value_range));
        }

        if (param.x2_bcast && param.x3_bcast) {
          uint32_t x2_index = i % param.x2_size;
          uint32_t x3_index = i % param.x3_size;
          param.exp[k * param.y_stride + i] = param.x1[k * param.x1_stride + i] == 1
                                                  ? static_cast<T>(param.x2_[x2_index])
                                                  : static_cast<T>(param.x3_[x3_index]);
        } else if (param.x2_bcast) {
          uint32_t x2_index = i % param.x2_size;
          uint32_t x3_index = k * param.x3_stride + i;
          param.exp[k * param.y_stride + i] =
              param.x1[k * param.x1_stride + i] == 1 ? static_cast<T>(param.x2_[x2_index]) : param.x3[x3_index];
        } else if (param.x3_bcast) {
          uint32_t x2_index = k * param.x2_stride + i;
          uint32_t x3_index = i % param.x3_size;
          param.exp[k * param.y_stride + i] =
              param.x1[k * param.x1_stride + i] == 1 ? param.x2[x2_index] : static_cast<T>(param.x3_[x3_index]);
        } else {
          uint32_t x2_index = k * param.x2_stride + i;
          uint32_t x3_index = k * param.x3_stride + i;
          param.exp[k * param.y_stride + i] =
              param.x1[k * param.x1_stride + i] == 1 ? param.x2[x2_index] : param.x3[x3_index];
        }
      }
    }
  }

  static void CreateNormalInput(WhereInputParam<int64_t> &param) {
    // 构造测试输入和预期结果
    uint32_t y_align = (param.size * sizeof(int64_t) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    uint32_t x1_align = (param.size * sizeof(uint8_t) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    uint32_t x2_align = (param.x2_size * sizeof(int64_t) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    uint32_t x3_align = (param.x3_size * sizeof(int64_t) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;

    param.y_stride = y_align / sizeof(int64_t);
    param.x1_stride = x1_align / sizeof(uint8_t);
    param.x2_stride = x2_align / sizeof(int64_t);
    param.x3_stride = x3_align / sizeof(int64_t);

    param.y = (int64_t *)AscendC::GmAlloc(y_align * param.m);
    param.x1 = (uint8_t *)AscendC::GmAlloc(x1_align * param.m);
    param.x2 = (int64_t *)AscendC::GmAlloc(x2_align * param.m);
    param.x3 = (int64_t *)AscendC::GmAlloc(x3_align * param.m);
    param.exp = (int64_t *)AscendC::GmAlloc(y_align * param.m);
    int mask_value_range = 2;
    int input_value_range = 20000;
    int input_offset = 2;
    for (int k = 0; k < param.m; k++) {
      for (int i = 0; i < param.size; i++) {
        param.x1[k * param.x1_stride + i] = static_cast<uint8_t>(i % mask_value_range);
        if (!param.x2_bcast) {
          param.x2[k * param.x2_stride + i] = static_cast<int64_t>((i + input_offset) % input_value_range) + 20000001;
        } else if (i < param.x2_size) { /* scalar */
          param.x2[i] = static_cast<int64_t>((0 + input_offset) % input_value_range) + 20000001;
        }
        if (!param.x3_bcast) {
          param.x3[k * param.x3_stride + i] =
              static_cast<int64_t>(-((i + input_offset) % input_value_range)) - 20000001;
        } else if (i < param.x3_size) { /* scalar */
          param.x3[i] = static_cast<int64_t>(-((0 + input_offset) % input_value_range)) - 20000001;
        }

        uint32_t x2_index = k * param.x2_stride + i;
        uint32_t x3_index = k * param.x3_stride + i;
        if (param.x2_bcast && param.x3_bcast) {
          x2_index = i % param.x2_size;
          x3_index = i % param.x3_size;
        } else if (param.x2_bcast) {
          x2_index = i % param.x2_size;
        } else if (param.x3_bcast) {
          x3_index = i % param.x3_size;
        } else {
          x2_index = k * param.x2_stride + i;
          x3_index = k * param.x3_stride + i;
        }
        param.exp[k * param.y_stride + i] =
            param.x1[k * param.x1_stride + i] == 1 ? param.x2[x2_index] : param.x3[x3_index];
      }
    }
  }

  template <typename T>
  static uint32_t NormalValid(WhereInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t k = 0; k < param.m; k++) {
      for (uint32_t i = 0; i < param.size; i++) {
        if (!DefaultCompare(param.y[k * param.y_stride + i], param.exp[k * param.y_stride + i])) {
          diff_count++;
        }
      }
    }
    return diff_count;
  }

  template <typename T>
  static void InvokeNormalKernel(WhereInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, x3buf, ybuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(uint8_t) * param.x1_stride * param.m);
    tpipe.InitBuffer(x2buf, param.x2_bcast ? ONE_BLK_SIZE : sizeof(T) * param.x2_stride * param.m);
    tpipe.InitBuffer(x3buf, param.x3_bcast ? ONE_BLK_SIZE : sizeof(T) * param.x3_stride * param.m);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.y_stride * param.m);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<uint8_t> l_x1 = x1buf.Get<uint8_t>();
    LocalTensor<T> l_x2 = x2buf.Get<T>();
    LocalTensor<T> l_x3 = x3buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();
    LocalTensor<float> l_x2_ = x2buf.Get<float>();
    LocalTensor<float> l_x3_ = x3buf.Get<float>();

    GmToUb<uint8_t>(l_x1, param.x1, param.x1_stride * param.m);
    if (param.x2_bcast) {
      GmToUb(l_x2_, param.x2_, param.x2_stride);
    } else {
      GmToUb(l_x2, param.x2, param.x2_stride * param.m);
    }
    if (param.x3_bcast) {
      GmToUb(l_x3_, param.x3_, param.x3_stride);
    } else {
      GmToUb(l_x3, param.x3, param.x3_stride * param.m);
    }
    GmToUb(l_y, param.y, param.y_stride * param.m);

    if (param.x2_bcast && param.x3_bcast) {
      Where<true, true>(l_y, l_x1, l_x2_, l_x3_, param.m, param.size, param.y_stride, param.x1_stride, param.x2_stride,
                        param.x3_stride, l_tmp, ONE_BLK_SIZE * 2);
    } else if (param.x2_bcast) {
      Where<true, false>(l_y, l_x1, l_x2_, l_x3, param.m, param.size, param.y_stride, param.x1_stride, param.x2_stride,
                         param.x3_stride, l_tmp, ONE_BLK_SIZE);
    } else if (param.x3_bcast) {
      Where<false, true>(l_y, l_x1, l_x2, l_x3_, param.m, param.size, param.y_stride, param.x1_stride, param.x2_stride,
                         param.x3_stride, l_tmp, ONE_BLK_SIZE);
    } else {
      Where<false, false>(l_y, l_x1, l_x2, l_x3, param.m, param.size, param.y_stride, param.x1_stride, param.x2_stride,
                          param.x3_stride, l_tmp, 0);
    }

    UbToGm(param.y, l_y, param.y_stride * param.m);
  }

  static void InvokeNormalKernel(WhereInputParam<int64_t> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, x3buf, ybuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(uint8_t) * param.x1_stride * param.m);
    tpipe.InitBuffer(x2buf, param.x2_bcast ? ONE_BLK_SIZE : sizeof(int64_t) * param.x2_stride * param.m);
    tpipe.InitBuffer(x3buf, param.x3_bcast ? ONE_BLK_SIZE : sizeof(int64_t) * param.x3_stride * param.m);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * param.y_stride * param.m);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<uint8_t> l_x1 = x1buf.Get<uint8_t>();
    LocalTensor<int64_t> l_x2 = x2buf.Get<int64_t>();
    LocalTensor<int64_t> l_x3 = x3buf.Get<int64_t>();
    LocalTensor<int64_t> l_y = ybuf.Get<int64_t>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb<uint8_t>(l_x1, param.x1, param.x1_stride * param.m);
    if (param.x2_bcast) {
      GmToUb(l_x2, param.x2, param.x2_stride);
    } else {
      GmToUb(l_x2, param.x2, param.x2_stride * param.m);
    }
    if (param.x3_bcast) {
      GmToUb(l_x3, param.x3, param.x3_stride);
    } else {
      GmToUb(l_x3, param.x3, param.x3_stride * param.m);
    }
    GmToUb(l_y, param.y, param.y_stride * param.m);

    if (param.x2_bcast && param.x3_bcast) {
      Where<true, true>(l_y, l_x1, l_x2, l_x3, param.m, param.size, param.y_stride, param.x1_stride, param.x2_stride,
                        param.x3_stride, l_tmp, ONE_BLK_SIZE * 2);
    } else if (param.x2_bcast) {
      Where<true, false>(l_y, l_x1, l_x2, l_x3, param.m, param.size, param.y_stride, param.x1_stride, param.x2_stride,
                         param.x3_stride, l_tmp, ONE_BLK_SIZE);
    } else if (param.x3_bcast) {
      Where<false, true>(l_y, l_x1, l_x2, l_x3, param.m, param.size, param.y_stride, param.x1_stride, param.x2_stride,
                         param.x3_stride, l_tmp, ONE_BLK_SIZE);
    } else {
      Where<false, false>(l_y, l_x1, l_x2, l_x3, param.m, param.size, param.y_stride, param.x1_stride, param.x2_stride,
                          param.x3_stride, l_tmp, 0);
    }

    UbToGm(param.y, l_y, param.y_stride * param.m);
  }

  template <typename T>
  static void WhereNormalTest(uint32_t m, uint32_t n, uint32_t x2_n = 0, uint32_t x3_n = 0) {
    WhereInputParam<T> param{};
    param.m = m;
    param.size = n;
    param.x2_size = x2_n == 0 ? n : /* scalar */ ONE_BLK_SIZE / sizeof(float);
    param.x2_bcast = x2_n == 0 ? false : /* scalar */ true;
    param.x3_size = x3_n == 0 ? n : /* scalar */ ONE_BLK_SIZE / sizeof(float);
    param.x3_bcast = x3_n == 0 ? false : /* scalar */ true;

    CreateNormalInput(param);

    // 构造Api调用函数
    auto kernel = [&param]() { InvokeNormalKernel<T>(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = NormalValid(param);
    EXPECT_EQ(diff_count, 0);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.x1);
    AscendC::GmFree(param.x2);
    AscendC::GmFree(param.x3);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.x2_);
    AscendC::GmFree(param.x3_);
  }

  static void WhereNormalTest(uint32_t m, uint32_t n, uint32_t x2_n = 0, uint32_t x3_n = 0) {
    WhereInputParam<int64_t> param{};
    param.m = m;
    param.size = n;
    param.x2_size = x2_n == 0 ? n : /* scalar */ ONE_BLK_SIZE / sizeof(int64_t);
    param.x2_bcast = x2_n == 0 ? false : /* scalar */ true;
    param.x3_size = x3_n == 0 ? n : /* scalar */ ONE_BLK_SIZE / sizeof(int64_t);
    param.x3_bcast = x3_n == 0 ? false : /* scalar */ true;

    CreateNormalInput(param);

    // 构造Api调用函数
    auto kernel = [&param]() { InvokeNormalKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = NormalValid(param);
    EXPECT_EQ(diff_count, 0);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.x1);
    AscendC::GmFree(param.x2);
    AscendC::GmFree(param.x3);
    AscendC::GmFree(param.exp);
  }
};

// 场景1
TEST_F(TestApiWhereUT, Where_X2S_X3S_float) {
  WhereTest<float>(ONE_BLK_SIZE / sizeof(float), 1, 1);
  WhereTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), 1, 1);
  WhereTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), 1, 1);
  WhereTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 1, 1);
  WhereTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 1, 1);
  WhereTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), 1, 1);
  WhereTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                    (ONE_BLK_SIZE - sizeof(float))) /
                       sizeof(float),
                   1, 1);
}
TEST_F(TestApiWhereUT, Where_X2S_X3S_half) {
  WhereTest<half>(ONE_BLK_SIZE / sizeof(half), 1, 1);
  WhereTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, 1);
  WhereTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, 1);
  WhereTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 1, 1);
  WhereTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, 1);
  WhereTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 1, 1);
  WhereTest<half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                   (ONE_BLK_SIZE - sizeof(half))) /
                      sizeof(half),
                  1, 1);
}
TEST_F(TestApiWhereUT, Where_X2S_X3S_int64) {
  WhereTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t), 1, 1);
  WhereTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 1);
  WhereTest<int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 1);
  WhereTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 1, 1);
  WhereTest<int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 1);
  WhereTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 1, 1);
  WhereTest<int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                      (ONE_BLK_SIZE - sizeof(int64_t))) /
                         sizeof(int64_t),
                     1, 1);
}
TEST_F(TestApiWhereUT, Where_X2S_X3S_int32) {
  WhereTest<int32_t>(ONE_BLK_SIZE / sizeof(int32_t), 1, 1);
  WhereTest<int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1, 1);
  WhereTest<int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1, 1);
  WhereTest<int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 1, 1);
  WhereTest<int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 1, 1);
  WhereTest<int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1, 1);
  WhereTest<int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                      (ONE_BLK_SIZE - sizeof(int32_t))) /
                         sizeof(int32_t),
                     1, 1);
}
TEST_F(TestApiWhereUT, Where_X2S_X3S_int16) {
  WhereTest<int16_t>(ONE_BLK_SIZE / sizeof(int16_t), 1, 1);
  WhereTest<int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1, 1);
  WhereTest<int16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1, 1);
  WhereTest<int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 1, 1);
  WhereTest<int16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1, 1);
  WhereTest<int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 1, 1);
  WhereTest<int16_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                      (ONE_BLK_SIZE - sizeof(int16_t))) /
                         sizeof(int16_t),
                     1, 1);
}

// 场景2
TEST_F(TestApiWhereUT, Where_X2S_float) {
  WhereTest<float>(ONE_BLK_SIZE / sizeof(float), 1);
  WhereTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), 1);
  WhereTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), 1);
  WhereTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 1);
  WhereTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 1);
  WhereTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), 1);
  WhereTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                    (ONE_BLK_SIZE - sizeof(float))) /
                       sizeof(float),
                   1);
}
TEST_F(TestApiWhereUT, Where_X2S_half) {
  WhereTest<half>(ONE_BLK_SIZE / sizeof(half), 1);
  WhereTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), 1);
  WhereTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), 1);
  WhereTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 1);
  WhereTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), 1);
  WhereTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 1);
  WhereTest<half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                   (ONE_BLK_SIZE - sizeof(half))) /
                      sizeof(half),
                  1);
}
TEST_F(TestApiWhereUT, Where_X2S_int64) {
  WhereTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t), 1);
  WhereTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1);
  WhereTest<int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1);
  WhereTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 1);
  WhereTest<int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1);
  WhereTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 1);
  WhereTest<int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                      (ONE_BLK_SIZE - sizeof(int64_t))) /
                         sizeof(int64_t),
                     1);
}
TEST_F(TestApiWhereUT, Where_X2S_int32) {
  WhereTest<int32_t>(ONE_BLK_SIZE / sizeof(int32_t), 1);
  WhereTest<int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1);
  WhereTest<int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1);
  WhereTest<int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 1);
  WhereTest<int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 1);
  WhereTest<int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1);
  WhereTest<int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                      (ONE_BLK_SIZE - sizeof(int32_t))) /
                         sizeof(int32_t),
                     1);
}
TEST_F(TestApiWhereUT, Where_X2S_int16) {
  WhereTest<int16_t>(ONE_BLK_SIZE / sizeof(int16_t), 1);
  WhereTest<int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1);
  WhereTest<int16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1);
  WhereTest<int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 1);
  WhereTest<int16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1);
  WhereTest<int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 1);
  WhereTest<int16_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                      (ONE_BLK_SIZE - sizeof(int16_t))) /
                         sizeof(int16_t),
                     1);
}

// 场景3
TEST_F(TestApiWhereUT, Where_X3S_float) {
  WhereTest<float>(ONE_BLK_SIZE / sizeof(float), 0, 1);
  WhereTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), 0, 1);
  WhereTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), 0, 1);
  WhereTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 0, 1);
  WhereTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 0, 1);
  WhereTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), 0, 1);
  WhereTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                    (ONE_BLK_SIZE - sizeof(float))) /
                       sizeof(float),
                   0, 1);
}
TEST_F(TestApiWhereUT, Where_X3S_half) {
  WhereTest<half>(ONE_BLK_SIZE / sizeof(half), 0, 1);
  WhereTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half), 0, 1);
  WhereTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), 0, 1);
  WhereTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 0, 1);
  WhereTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), 0, 1);
  WhereTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 0, 1);
  WhereTest<half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                   (ONE_BLK_SIZE - sizeof(half))) /
                      sizeof(half),
                  0, 1);
}
TEST_F(TestApiWhereUT, Where_X3S_int64) {
  WhereTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t), 0, 1);
  WhereTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 0, 1);
  WhereTest<int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 0, 1);
  WhereTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 0, 1);
  WhereTest<int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 0, 1);
  WhereTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 0, 1);
  WhereTest<int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                      (ONE_BLK_SIZE - sizeof(int64_t))) /
                         sizeof(int64_t),
                     0, 1);
}
TEST_F(TestApiWhereUT, Where_X3S_int32) {
  WhereTest<int32_t>(ONE_BLK_SIZE / sizeof(int32_t), 0, 1);
  WhereTest<int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 0, 1);
  WhereTest<int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 0, 1);
  WhereTest<int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 0, 1);
  WhereTest<int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 0, 1);
  WhereTest<int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 0, 1);
  WhereTest<int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                      (ONE_BLK_SIZE - sizeof(int32_t))) /
                         sizeof(int32_t),
                     0, 1);
}
TEST_F(TestApiWhereUT, Where_X3S_int16) {
  WhereTest<int16_t>(ONE_BLK_SIZE / sizeof(int16_t), 0, 1);
  WhereTest<int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 0, 1);
  WhereTest<int16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 0, 1);
  WhereTest<int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 0, 1);
  WhereTest<int16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 0, 1);
  WhereTest<int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 0, 1);
  WhereTest<int16_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                      (ONE_BLK_SIZE - sizeof(int16_t))) /
                         sizeof(int16_t),
                     0, 1);
}

// 场景4
TEST_F(TestApiWhereUT, Where_float) {
  WhereTest<float>(ONE_BLK_SIZE / sizeof(float));
  WhereTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  WhereTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  WhereTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  // todo 因输入多，总buffer不够，导致用例无法执行
  //  WhereTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float));
  //  WhereTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float));
  //  WhereTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
  //                    (ONE_BLK_SIZE - sizeof(float))) /
  //                   sizeof(float));
}

TEST_F(TestApiWhereUT, Where_half) {
  WhereTest<half>(ONE_BLK_SIZE / sizeof(half));
  WhereTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  WhereTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  WhereTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  // todo 因输入多，总buffer不够，导致用例无法执行
  //   WhereTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half));
  //   WhereTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half));
  //   WhereTest<half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
  //                    (ONE_BLK_SIZE - sizeof(half))) /
  //                       sizeof(half));
}
TEST_F(TestApiWhereUT, Where_int64) {
  WhereTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t));
  WhereTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  WhereTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t));
  WhereTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t));
  // todo 因输入多，总buffer不够，导致用例无法执行
  //   WhereTest<int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  //   WhereTest<int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  //   WhereTest<int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
  //                       (ONE_BLK_SIZE - sizeof(int64_t))) /
  //                          sizeof(int64_t));
}
TEST_F(TestApiWhereUT, Where_int32) {
  WhereTest<int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  WhereTest<int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  WhereTest<int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  WhereTest<int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  // todo 因输入多，总buffer不够，导致用例无法执行
  //   WhereTest<int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  //   WhereTest<int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  //   WhereTest<int32_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
  //                       (ONE_BLK_SIZE - sizeof(int32_t))) /
  //                          sizeof(int32_t));
}
TEST_F(TestApiWhereUT, Where_int16) {
  WhereTest<int16_t>(ONE_BLK_SIZE / sizeof(int16_t));
  WhereTest<int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  WhereTest<int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
  WhereTest<int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t));
  // todo 因输入多，总buffer不够，导致用例无法执行
  //   WhereTest<int16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  //   WhereTest<int16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  //   WhereTest<int16_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
  //                       (ONE_BLK_SIZE - sizeof(int16_t))) /
  //                          sizeof(int16_t));
}

// Normal
// 场景1
TEST_F(TestApiWhereUT, Where_X2S_X3S_float_normal) {
  WhereNormalTest<float>(1, ONE_BLK_SIZE / sizeof(float), 1, 1);
  WhereNormalTest<float>(1, ONE_REPEAT_BYTE_SIZE / sizeof(float), 1, 1);
  WhereNormalTest<float>(1, (ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 1, 1);
  WhereNormalTest<float>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 1, 1);

  WhereNormalTest<float>(71, ONE_BLK_SIZE / sizeof(float), 1, 1);
  WhereNormalTest<float>(71, ONE_REPEAT_BYTE_SIZE / sizeof(float), 1, 1);
  WhereNormalTest<float>(71, (ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 1, 1);
  WhereNormalTest<float>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 1, 1);
}
TEST_F(TestApiWhereUT, Where_X2S_X3S_half_normal) {
  WhereNormalTest<half>(1, ONE_BLK_SIZE / sizeof(half), 1, 1);
  WhereNormalTest<half>(1, ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, 1);
  WhereNormalTest<half>(1, (ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 1, 1);
  WhereNormalTest<half>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 1, 1);

  WhereNormalTest<half>(3, ONE_BLK_SIZE / sizeof(half), 1, 1);
  WhereNormalTest<half>(4, ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, 1);
  WhereNormalTest<half>(5, (ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 1, 1);
  WhereNormalTest<half>(6, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 1, 1);

  WhereNormalTest<half>(15, ONE_BLK_SIZE / sizeof(half), 1, 1);
  WhereNormalTest<half>(16, ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, 1);
  WhereNormalTest<half>(17, (ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 1, 1);
  WhereNormalTest<half>(18, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 1, 1);

  WhereNormalTest<half>(63, ONE_BLK_SIZE / sizeof(half), 1, 1);
  WhereNormalTest<half>(64, ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, 1);
  WhereNormalTest<half>(65, (ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 1, 1);
  WhereNormalTest<half>(66, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 1, 1);
}
TEST_F(TestApiWhereUT, Where_X2S_X3S_int64_normal) {
  WhereNormalTest(1, ONE_BLK_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTest(1, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTest(1, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 1, 1);
  WhereNormalTest(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 1, 1);

  WhereNormalTest(3, ONE_BLK_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTest(4, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTest(5, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 1, 1);
  WhereNormalTest(6, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 1, 1);

  WhereNormalTest(15, ONE_BLK_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTest(16, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTest(17, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 1, 1);
  WhereNormalTest(18, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 1, 1);

  WhereNormalTest(63, ONE_BLK_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTest(64, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTest(65, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 1, 1);
  WhereNormalTest(66, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 1, 1);
}
TEST_F(TestApiWhereUT, Where_X2S_X3S_int32_normal) {
  WhereNormalTest<int32_t>(1, ONE_BLK_SIZE / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(1, (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 1, 1);

  WhereNormalTest<int32_t>(3, ONE_BLK_SIZE / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(4, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(5, (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(6, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 1, 1);

  WhereNormalTest<int32_t>(15, ONE_BLK_SIZE / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(16, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(17, (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(18, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 1, 1);

  WhereNormalTest<int32_t>(63, ONE_BLK_SIZE / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(64, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(65, (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 1, 1);
  WhereNormalTest<int32_t>(66, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 1, 1);
}
TEST_F(TestApiWhereUT, Where_X2S_X3S_int16_normal) {
  WhereNormalTest<int16_t>(1, ONE_BLK_SIZE / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(1, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 1, 1);

  WhereNormalTest<int16_t>(3, ONE_BLK_SIZE / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(4, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(5, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(6, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 1, 1);

  WhereNormalTest<int16_t>(15, ONE_BLK_SIZE / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(16, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(17, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(18, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 1, 1);

  WhereNormalTest<int16_t>(63, ONE_BLK_SIZE / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(64, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(65, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 1, 1);
  WhereNormalTest<int16_t>(66, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 1, 1);
}

// 场景2
TEST_F(TestApiWhereUT, Where_X2S_float_normal) {
  WhereNormalTest<float>(1, ONE_BLK_SIZE / sizeof(float), 1, 0);
  WhereNormalTest<float>(1, ONE_REPEAT_BYTE_SIZE / sizeof(float), 1, 0);
  WhereNormalTest<float>(1, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), 1, 0);
  WhereNormalTest<float>(1, (ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 1, 0);
  WhereNormalTest<float>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 1, 0);
  WhereNormalTest<float>(1, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), 1, 0);
  WhereNormalTest<float>(1,
                         ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                          (ONE_BLK_SIZE - sizeof(float))) /
                             sizeof(float),
                         1, 0);

  WhereNormalTest<float>(71, ONE_BLK_SIZE / sizeof(float), 1, 0);
  WhereNormalTest<float>(71, ONE_REPEAT_BYTE_SIZE / sizeof(float), 1, 0);
  WhereNormalTest<float>(71, (ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 1, 0);
  WhereNormalTest<float>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 1, 0);
}

TEST_F(TestApiWhereUT, Where_X2S_half_normal) {
  WhereNormalTest<half>(1, ONE_BLK_SIZE / sizeof(half), 1, 0);
  WhereNormalTest<half>(1, ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, 0);
  WhereNormalTest<half>(1, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, 0);
  WhereNormalTest<half>(1, (ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 1, 0);
  WhereNormalTest<half>(1, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, 0);
  WhereNormalTest<half>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 1, 0);
  WhereNormalTest<half>(1,
                        ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                         (ONE_BLK_SIZE - sizeof(half))) /
                            sizeof(half),
                        1, 0);

  WhereNormalTest<half>(71, ONE_BLK_SIZE / sizeof(half), 1, 0);
  WhereNormalTest<half>(71, ONE_REPEAT_BYTE_SIZE / sizeof(half), 1, 0);
  WhereNormalTest<half>(71, (ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 1, 0);
  WhereNormalTest<half>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 1, 0);
}
TEST_F(TestApiWhereUT, Where_X2S_int64_normal) {
  WhereNormalTest(1, ONE_BLK_SIZE / sizeof(int64_t), 1, 0);
  WhereNormalTest(1, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 0);
  WhereNormalTest(1, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 0);
  WhereNormalTest(1, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 1, 0);
  WhereNormalTest(1, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 0);
  WhereNormalTest(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 1, 0);
  WhereNormalTest(1,
                  ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                   (ONE_BLK_SIZE - sizeof(int64_t))) /
                      sizeof(int64_t),
                  1, 0);

  WhereNormalTest(71, ONE_BLK_SIZE / sizeof(int64_t), 1, 0);
  WhereNormalTest(71, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 0);
  WhereNormalTest(71, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 1, 0);
  WhereNormalTest(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 1, 0);
}
TEST_F(TestApiWhereUT, Where_X2S_int32_normal) {
  WhereNormalTest<int32_t>(1, ONE_BLK_SIZE / sizeof(int32_t), 1, 0);
  WhereNormalTest<int32_t>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1, 0);
  WhereNormalTest<int32_t>(1, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1, 0);
  WhereNormalTest<int32_t>(1, (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 1, 0);
  WhereNormalTest<int32_t>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 1, 0);
  WhereNormalTest<int32_t>(1, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1, 0);
  WhereNormalTest<int32_t>(1,
                           ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                            (ONE_BLK_SIZE - sizeof(int32_t))) /
                               sizeof(int32_t),
                           1, 0);

  WhereNormalTest<int32_t>(71, ONE_BLK_SIZE / sizeof(int32_t), 1, 0);
  WhereNormalTest<int32_t>(71, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 1, 0);
  WhereNormalTest<int32_t>(71, (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 1, 0);
  WhereNormalTest<int32_t>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 1, 0);
}
TEST_F(TestApiWhereUT, Where_X2S_int16_normal) {
  WhereNormalTest<int16_t>(1, ONE_BLK_SIZE / sizeof(int16_t), 1, 0);
  WhereNormalTest<int16_t>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1, 0);
  WhereNormalTest<int16_t>(1, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1, 0);
  WhereNormalTest<int16_t>(1, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 1, 0);
  WhereNormalTest<int16_t>(1, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1, 0);
  WhereNormalTest<int16_t>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 1, 0);
  WhereNormalTest<int16_t>(1,
                           ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                            (ONE_BLK_SIZE - sizeof(int16_t))) /
                               sizeof(int16_t),
                           1, 0);

  WhereNormalTest<int16_t>(71, ONE_BLK_SIZE / sizeof(int16_t), 1, 0);
  WhereNormalTest<int16_t>(71, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 1, 0);
  WhereNormalTest<int16_t>(71, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 1, 0);
  WhereNormalTest<int16_t>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 1, 0);
}

// 场景3
TEST_F(TestApiWhereUT, Where_X3S_float_normal) {
  WhereNormalTest<float>(1, ONE_BLK_SIZE / sizeof(float), 0, 1);
  WhereNormalTest<float>(1, ONE_REPEAT_BYTE_SIZE / sizeof(float), 0, 1);
  WhereNormalTest<float>(1, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float), 0, 1);
  WhereNormalTest<float>(1, (ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 0, 1);
  WhereNormalTest<float>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 0, 1);
  WhereNormalTest<float>(1, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float), 0, 1);
  WhereNormalTest<float>(1,
                         ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                          (ONE_BLK_SIZE - sizeof(float))) /
                             sizeof(float),
                         0, 1);

  WhereNormalTest<float>(71, ONE_BLK_SIZE / sizeof(float), 0, 1);
  WhereNormalTest<float>(71, ONE_REPEAT_BYTE_SIZE / sizeof(float), 0, 1);
  WhereNormalTest<float>(71, (ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 0, 1);
  WhereNormalTest<float>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 0, 1);
}
TEST_F(TestApiWhereUT, Where_X3S_half_normal) {
  WhereNormalTest<half>(1, ONE_BLK_SIZE / sizeof(half), 0, 1);
  WhereNormalTest<half>(1, ONE_REPEAT_BYTE_SIZE / sizeof(half), 0, 1);
  WhereNormalTest<half>(1, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(half), 0, 1);
  WhereNormalTest<half>(1, (ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 0, 1);
  WhereNormalTest<half>(1, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(half), 0, 1);
  WhereNormalTest<half>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 0, 1);
  WhereNormalTest<half>(1,
                        ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                         (ONE_BLK_SIZE - sizeof(half))) /
                            sizeof(half),
                        0, 1);

  WhereNormalTest<half>(71, ONE_BLK_SIZE / sizeof(half), 0, 1);
  WhereNormalTest<half>(71, ONE_REPEAT_BYTE_SIZE / sizeof(half), 0, 1);
  WhereNormalTest<half>(71, (ONE_BLK_SIZE - sizeof(half)) / sizeof(half), 0, 1);
  WhereNormalTest<half>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half), 0, 1);
}
TEST_F(TestApiWhereUT, Where_X3S_int64_normal) {
  WhereNormalTest(1, ONE_BLK_SIZE / sizeof(int64_t), 0, 1);
  WhereNormalTest(1, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 0, 1);
  WhereNormalTest(1, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 0, 1);
  WhereNormalTest(1, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 0, 1);
  WhereNormalTest(1, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 0, 1);
  WhereNormalTest(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 0, 1);
  WhereNormalTest(1,
                  ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                   (ONE_BLK_SIZE - sizeof(int64_t))) /
                      sizeof(int64_t),
                  0, 1);

  WhereNormalTest(71, ONE_BLK_SIZE / sizeof(int64_t), 0, 1);
  WhereNormalTest(71, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 0, 1);
  WhereNormalTest(71, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 0, 1);
  WhereNormalTest(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 0, 1);
}
TEST_F(TestApiWhereUT, Where_X3S_int32_normal) {
  WhereNormalTest<int32_t>(1, ONE_BLK_SIZE / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t>(1, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t>(1, (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t>(1, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t>(1,
                           ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                            (ONE_BLK_SIZE - sizeof(int32_t))) /
                               sizeof(int32_t),
                           0, 1);

  WhereNormalTest<int32_t>(71, ONE_BLK_SIZE / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t>(71, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t>(71, (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 0, 1);
}
TEST_F(TestApiWhereUT, Where_X3S_int16_normal) {
  WhereNormalTest<int16_t>(1, ONE_BLK_SIZE / sizeof(int16_t), 0, 1);
  WhereNormalTest<int16_t>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 0, 1);
  WhereNormalTest<int16_t>(1, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 0, 1);
  WhereNormalTest<int16_t>(1, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 0, 1);
  WhereNormalTest<int16_t>(1, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 0, 1);
  WhereNormalTest<int16_t>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 0, 1);
  WhereNormalTest<int16_t>(1,
                           ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                            (ONE_BLK_SIZE - sizeof(int16_t))) /
                               sizeof(int16_t),
                           0, 1);

  WhereNormalTest<int16_t>(71, ONE_BLK_SIZE / sizeof(int16_t), 0, 1);
  WhereNormalTest<int16_t>(71, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t), 0, 1);
  WhereNormalTest<int16_t>(71, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t), 0, 1);
  WhereNormalTest<int16_t>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t), 0, 1);
}

// 场景4
TEST_F(TestApiWhereUT, Where_float_normal) {
  WhereNormalTest<float>(1, ONE_BLK_SIZE / sizeof(float));
  WhereNormalTest<float>(1, ONE_REPEAT_BYTE_SIZE / sizeof(float));
  WhereNormalTest<float>(1, (ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  WhereNormalTest<float>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));

  WhereNormalTest<float>(71, ONE_BLK_SIZE / sizeof(float));
  WhereNormalTest<float>(71, ONE_REPEAT_BYTE_SIZE / sizeof(float));
  WhereNormalTest<float>(71, (ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  WhereNormalTest<float>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
}
TEST_F(TestApiWhereUT, Where_half_normal) {
  WhereNormalTest<half>(1, ONE_BLK_SIZE / sizeof(half));
  WhereNormalTest<half>(1, ONE_REPEAT_BYTE_SIZE / sizeof(half));
  WhereNormalTest<half>(1, (ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  WhereNormalTest<half>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));

  WhereNormalTest<half>(71, ONE_BLK_SIZE / sizeof(half));
  WhereNormalTest<half>(71, ONE_REPEAT_BYTE_SIZE / sizeof(half));
  WhereNormalTest<half>(71, (ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  WhereNormalTest<half>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
}

TEST_F(TestApiWhereUT, Where_int64_normal) {
  WhereNormalTest(1, ONE_BLK_SIZE / sizeof(int64_t));
  WhereNormalTest(1, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  WhereNormalTest(1, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t));
  WhereNormalTest(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t));

  WhereNormalTest(71, ONE_BLK_SIZE / sizeof(int64_t));
  WhereNormalTest(71, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  WhereNormalTest(71, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t));
  WhereNormalTest(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t));
}

TEST_F(TestApiWhereUT, Where_int32_normal) {
  WhereNormalTest<int32_t>(1, ONE_BLK_SIZE / sizeof(int32_t));
  WhereNormalTest<int32_t>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  WhereNormalTest<int32_t>(1, (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  WhereNormalTest<int32_t>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));

  WhereNormalTest<int32_t>(71, ONE_BLK_SIZE / sizeof(int32_t));
  WhereNormalTest<int32_t>(71, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  WhereNormalTest<int32_t>(71, (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  WhereNormalTest<int32_t>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
}
TEST_F(TestApiWhereUT, Where_int16_normal) {
  WhereNormalTest<int16_t>(1, ONE_BLK_SIZE / sizeof(int16_t));
  WhereNormalTest<int16_t>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  WhereNormalTest<int16_t>(1, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
  WhereNormalTest<int16_t>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t));

  WhereNormalTest<int16_t>(71, ONE_BLK_SIZE / sizeof(int16_t));
  WhereNormalTest<int16_t>(71, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  WhereNormalTest<int16_t>(71, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
  WhereNormalTest<int16_t>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t));
}
// ST-7：真机 vsel 谓词位序探针——用固定模式谓词观察 Select 按 word 的选择结果，
// 钉死谓词 bit 与 32bit word 的映射关系。
TEST_F(TestApiWhereUT, ST7_SelectPredicateBitOrder) {
  constexpr uint32_t kLanes = 64;  // float lanes = 32bit words
  auto *s0 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kLanes));
  auto *s1 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kLanes));
  auto *out = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kLanes));
  auto *pred_dbg = static_cast<uint8_t *>(AscendC::GmAlloc(16));
  for (uint32_t i = 0; i < kLanes; i++) {
    s0[i] = 100.0f + i;   // src0 标识：100+i
    s1[i] = -100.0f - i;  // src1 标识：-100-i
  }
  auto kernel = [s0, s1, out, pred_dbg] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(float) * kLanes);
    tpipe.InitBuffer(x2buf, sizeof(float) * kLanes);
    tpipe.InitBuffer(ybuf, sizeof(float) * kLanes);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);
    auto a = x1buf.Get<float>();
    auto b = x2buf.Get<float>();
    auto y = ybuf.Get<float>();
    auto scratch = tmp.Get<uint8_t>();
    for (uint32_t i = 0; i < kLanes; i++) {
      a.SetValue(i, 100.0f + i);
      b.SetValue(i, -100.0f - i);
    }
    // 谓词：0x33 0x33 ... = bit 模式 00110011...
    // 若谓词按 word 1bit：word j 的谓词 = bit j → word 2,3,6,7,10,11,... 选 src1，其余 src0
    // 若按 2bit/word 或其他映射，输出模式不同，据此判定真实位序
    LocalTensor<uint8_t> pred = scratch[0].template ReinterpretCast<uint8_t>();
    pred.SetSize(16);
    for (uint32_t i = 0; i < 16; i++) {
      pred.SetValue(i, 0x33);
    }
    UbToGm(pred_dbg, pred, 16);
    const BinaryRepeatParams rpt = {1, 1, 1, 8, 8, 8};
    Select<float, uint8_t, true>(y, pred, a, b, SELMODE::VSEL_TENSOR_TENSOR_MODE, kLanes, 1, rpt);
    UbToGm(out, y, kLanes);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  printf("[ST7] pred:");
  for (uint32_t i = 0; i < 4; i++) printf(" %02X", pred_dbg[i]);
  printf("\n[ST7] out:");
  for (uint32_t i = 0; i < 32; i++) printf(" %.0f", out[i]);
  printf("\n");
  // 谓词位序（真机实测）：word j ↔ 字节 j/8 的 bit 7-(j%8)，即字节内高位在前
  uint32_t diff = 0;
  for (uint32_t i = 0; i < kLanes; i++) {
    const uint8_t byte = pred_dbg[i / 8];
    const bool bit = ((byte >> (7 - (i % 8))) & 1U) != 0;
    const float exp = bit ? -100.0f - i : 100.0f + i;
    if (out[i] != exp) {
      diff++;
    }
  }
  EXPECT_EQ(diff, 0);
  AscendC::GmFree(s0);
  AscendC::GmFree(s1);
  AscendC::GmFree(out);
  AscendC::GmFree(pred_dbg);
}

// ST-8：谓词 word->bit 映射测绘——逐 bit 置位观察每个 word 的选择，一次性输出完整映射表。
TEST_F(TestApiWhereUT, ST8_PredicateWordBitMap) {
  constexpr uint32_t kLanes = 64;
  auto *s0 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kLanes));
  auto *s1 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kLanes));
  auto *out = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kLanes));
  auto *pred_dbg = static_cast<uint8_t *>(AscendC::GmAlloc(16));
  auto *bit2word = static_cast<uint32_t *>(AscendC::GmAlloc(sizeof(uint32_t) * 64));  // bit -> word
  for (uint32_t i = 0; i < kLanes; i++) {
    s0[i] = 1.0f;         // src0 标识
    s1[i] = 1000.0f + i;  // src1 标识 = 该 word 被"选中 src1"时可见 word 序号
  }
  for (uint32_t w = 0; w < 64; w++) {
    bit2word[w] = UINT32_MAX;
  }
  auto kernel = [s0, s1, out, pred_dbg] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(float) * kLanes);
    tpipe.InitBuffer(x2buf, sizeof(float) * kLanes);
    tpipe.InitBuffer(ybuf, sizeof(float) * kLanes);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);
    auto a = x1buf.Get<float>();
    auto b = x2buf.Get<float>();
    auto y = ybuf.Get<float>();
    auto scratch = tmp.Get<uint8_t>();
    for (uint32_t i = 0; i < kLanes; i++) {
      a.SetValue(i, 1.0f);
      b.SetValue(i, 1000.0f + i);
    }
    LocalTensor<uint8_t> pred = scratch[0].template ReinterpretCast<uint8_t>();
    pred.SetSize(16);
    for (uint32_t i = 0; i < 16; i++) {
      pred.SetValue(i, 0);
    }
    const BinaryRepeatParams rpt = {1, 1, 1, 8, 8, 8};
    Select<float, uint8_t, true>(y, pred, a, b, SELMODE::VSEL_TENSOR_TENSOR_MODE, kLanes, 1, rpt);
    UbToGm(out, y, kLanes);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  // 全 0 基线
  ICPU_RUN_KF(kernel, 1);
  printf("[ST8] baseline out[0]=%.0f\n", out[0]);

  // 逐 bit 置位：bit 0..63，观察哪个 word 变成 src1
  for (uint32_t bit = 0; bit < 64; bit++) {
    // 写谓词缓冲：只置 bit
    // bit 位于字节 bit/8 的 1<<(bit%8)（LSB-first 探测；MSB 由结果反推）
    // 直接在 GmToUb 前 reset pred：
    // 由于 kernel 里 pred 每次 run 会重置，这里改用 GM 传入
    // 简化：把 pred 构造移出 kernel——重写 kernel 内联
    auto kernel_bit = [bit, out, s0, s1]() {
      TPipe tpipe;
      TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;
      tpipe.InitBuffer(x1buf, sizeof(float) * kLanes);
      tpipe.InitBuffer(x2buf, sizeof(float) * kLanes);
      tpipe.InitBuffer(ybuf, sizeof(float) * kLanes);
      tpipe.InitBuffer(tmp, TMP_UB_SIZE);
      auto a = x1buf.Get<float>();
      auto b = x2buf.Get<float>();
      auto y = ybuf.Get<float>();
      auto scratch = tmp.Get<uint8_t>();
      for (uint32_t i = 0; i < kLanes; i++) {
        a.SetValue(i, 1.0f);
        b.SetValue(i, 1000.0f + i);
      }
      LocalTensor<uint8_t> pred = scratch[0].template ReinterpretCast<uint8_t>();
      pred.SetSize(16);
      for (uint32_t i = 0; i < 16; i++) {
        pred.SetValue(i, 0);
      }
      pred.SetValue(bit / 8, static_cast<uint8_t>(1U << (bit % 8)));
      const BinaryRepeatParams rpt = {1, 1, 1, 8, 8, 8};
      Select<float, uint8_t, true>(y, pred, a, b, SELMODE::VSEL_TENSOR_TENSOR_MODE, kLanes, 1, rpt);
      UbToGm(out, y, kLanes);
    };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel_bit, 1);
    // 被该 bit 置位而翻转为 src0 的 word（值 = 1.0 < 1000）；其它 word 保持 src1
    for (uint32_t w = 0; w < kLanes; w++) {
      if (out[w] < 1000.0f) {
        bit2word[bit] = w;
      }
    }
  }
  // 输出映射表：bit -> word
  printf("[ST8] bit->word map:");
  for (uint32_t bit = 0; bit < 64; bit++) {
    printf(" %u->%u", bit, bit2word[bit]);
  }
  printf("\n");
  // 全部 bit 应有唯一映射
  uint32_t unmapped = 0;
  for (uint32_t bit = 0; bit < 64; bit++) {
    if (bit2word[bit] == UINT32_MAX) {
      unmapped++;
    }
  }
  EXPECT_EQ(unmapped, 0);
  AscendC::GmFree(s0);
  AscendC::GmFree(s1);
  AscendC::GmFree(out);
  AscendC::GmFree(pred_dbg);
  AscendC::GmFree(bit2word);
}

// ST-10：WherePartitionBufferInt64 在 used_size > 0 时的容量校验复现。
// 现场问题（PR#3105）：load_mask_select_i64_store（1x171x306，MLIR e2e）在
// Where 前序算子占用 tmp（used_size=64B）后，分区函数把字节数当元素数传给
// LocalTensor<half>::SetSize，容量按 2 倍需求校验 → ICPU SetSize 越界 SIGABRT。
// 本用例以同样的 used_size 调用分区函数直接复现；修复（字节/sizeof(T) 换算）后通过。
TEST_F(TestApiWhereUT, ST10_WherePartitionInt64UsedSizeOverflow) {
  const uint32_t n = 306;                                                   // 与 e2e 用例的末轴一致
  const uint32_t mask_stride = AlignUp(n, ONE_BLK_SIZE / sizeof(uint8_t));  // 320
  auto *x1 = (uint8_t *)AscendC::GmAlloc(mask_stride);
  auto *x2 = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * n);
  auto *x3 = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * n);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * n);
  for (uint32_t i = 0; i < n; i++) {
    x1[i % 32] = (i % 2U == 0U) ? 1U : 0U;
    x2[i] = (int64_t)(0x1A2B3C4D5E6F7080LL - i);
    x3[i] = -(int64_t)(i + 1);
  }
  auto kernel = [x1, x2, x3, y, n] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1b, x2b, x3b, yb, tmp;
    tpipe.InitBuffer(x1b, 320);  // 覆盖 mask_stride（元素 n=306 对齐到 320）
    tpipe.InitBuffer(x2b, sizeof(int64_t) * n);
    tpipe.InitBuffer(x3b, sizeof(int64_t) * n);
    tpipe.InitBuffer(yb, sizeof(int64_t) * n);
    tpipe.InitBuffer(tmp, 4608);  // 与 e2e 相近的小 tmp，放大 used_size 的影响
    auto lx1 = x1b.Get<uint8_t>();
    auto lx2 = x2b.Get<int64_t>();
    auto lx3 = x3b.Get<int64_t>();
    auto ly = yb.Get<int64_t>();
    auto ltmp = tmp.Get<uint8_t>();
    for (uint32_t i = 0; i < n; i++) {
      lx1.SetValue(i, (i % 2U == 0U) ? 1U : 0U);
    }
    for (uint32_t i = 0; i < n; i++) {
      lx2.SetValue(i, (int64_t)(0x1A2B3C4D5E6F7080LL - i));
      lx3.SetValue(i, -(int64_t)(i + 1));
    }
    // used_size = 64：与 e2e 崩溃场景同构（前序算子占用 tmp 后再分区）
    const uint32_t data_stride = AlignUp(n, ONE_BLK_SIZE / sizeof(int64_t));  // 308
    Where<false, false>(ly, lx1, lx2, lx3, 1, n, data_stride, mask_stride, data_stride, data_stride, ltmp, 64);
    for (uint32_t i = 0; i < n; i++) {
      y[i] = ly.GetValue(i);
    }
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  for (uint32_t i = 0; i < n; i++) {
    const int64_t expected = (i % 2U == 0U) ? (int64_t)(0x1A2B3C4D5E6F7080LL - i) : -(int64_t)(i + 1);
    EXPECT_EQ(y[i], expected) << "index=" << i;
  }
  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(x3);
  AscendC::GmFree(y);
}

}  // namespace af
