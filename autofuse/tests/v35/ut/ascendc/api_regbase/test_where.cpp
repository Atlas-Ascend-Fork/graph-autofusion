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
#include "api_regbase/where.h"

using namespace AscendC;

namespace af {
template <typename T, uint8_t dim>
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
  T *x2_{};
  T *x3_{};
};

class TestRegbaseApiWhereUT : public testing::Test {
 protected:
  // normal
  template <typename T, uint8_t dim>
  static void CreateNormalInput(WhereInputParam<T, dim> &param) {
    // 构造测试输入和预期结果
    uint32_t y_align = (param.size * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    uint32_t x1_align = (param.size * sizeof(uint8_t) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    uint32_t x2_align =
        param.x2_bcast ? ONE_BLK_SIZE : (param.x2_size * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    uint32_t x3_align =
        param.x3_bcast ? ONE_BLK_SIZE : (param.x3_size * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;

    param.y_stride = y_align / sizeof(T);
    param.x1_stride = x1_align / sizeof(uint8_t);
    param.x2_stride = param.x2_bcast ? ONE_BLK_SIZE / sizeof(T) : x2_align / sizeof(T);
    param.x3_stride = param.x3_bcast ? ONE_BLK_SIZE / sizeof(T) : x3_align / sizeof(T);

    param.y = (T *)AscendC::GmAlloc(y_align * param.m);
    param.x1 = (uint8_t *)AscendC::GmAlloc(x1_align * param.m);
    param.x2 = (T *)AscendC::GmAlloc(x2_align * param.m);
    param.x3 = (T *)AscendC::GmAlloc(x3_align * param.m);
    param.exp = (T *)AscendC::GmAlloc(y_align * param.m);
    param.x2_ = (T *)AscendC::GmAlloc(x2_align);
    param.x3_ = (T *)AscendC::GmAlloc(x3_align);
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
  template <typename T, uint8_t dim>
  static void CreateNormalInputInt64(WhereInputParam<T, dim> &param) {
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

  template <typename T, uint8_t dim>
  static uint32_t NormalValid(WhereInputParam<T, dim> &param) {
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

  template <typename T, uint8_t dim>
  static void InvokeNormalKernel(WhereInputParam<T, dim> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, x3buf, ybuf;
    tpipe.InitBuffer(x1buf, sizeof(uint8_t) * param.x1_stride * param.m);
    tpipe.InitBuffer(x2buf, param.x2_bcast ? ONE_BLK_SIZE : sizeof(T) * param.x2_stride * param.m);
    tpipe.InitBuffer(x3buf, param.x3_bcast ? ONE_BLK_SIZE : sizeof(T) * param.x3_stride * param.m);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.y_stride * param.m);

    LocalTensor<uint8_t> l_x1 = x1buf.Get<uint8_t>();
    LocalTensor<T> l_x2 = x2buf.Get<T>();
    LocalTensor<T> l_x3 = x3buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();

    LocalTensor<T> l_x2_ = x2buf.Get<T>();
    LocalTensor<T> l_x3_ = x3buf.Get<T>();

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

    if constexpr (dim == 1) {
      const uint16_t output_dims[dim] = {(uint16_t)param.size};
      const uint16_t output_stride[dim] = {1};
      const uint16_t mask_stride[dim] = {1};
      const uint16_t input_stride[dim] = {1};

      if (param.x2_bcast && param.x3_bcast) {
        WhereExtend<true, true, 1, T, T, T>(l_y, l_x1, l_x2_, l_x3_, output_dims, output_stride, mask_stride,
                                            input_stride);
      } else if (param.x2_bcast) {
        WhereExtend<true, false, 1, T, T, T>(l_y, l_x1, l_x2_, l_x3, output_dims, output_stride, mask_stride,
                                             input_stride);
      } else if (param.x3_bcast) {
        WhereExtend<false, true, 1, T, T, T>(l_y, l_x1, l_x2, l_x3_, output_dims, output_stride, mask_stride,
                                             input_stride);
      } else {
        WhereExtend<false, false, 1, T, T, T>(l_y, l_x1, l_x2, l_x3, output_dims, output_stride, mask_stride,
                                              input_stride);
      }
    } else if constexpr (dim == 2) {
      const uint16_t output_dims[dim] = {(uint16_t)param.m, (uint16_t)param.size};
      const uint16_t output_stride[dim] = {(uint16_t)param.y_stride, 1};
      const uint16_t mask_stride[dim] = {(uint16_t)param.x1_stride, 1};
      const uint16_t input_stride[dim] = {(uint16_t)param.x2_stride, 1};

      if (param.x2_bcast && param.x3_bcast) {
        WhereExtend<true, true, 2, T, T, T>(l_y, l_x1, l_x2_, l_x3_, output_dims, output_stride, mask_stride,
                                            input_stride);
      } else if (param.x2_bcast) {
        WhereExtend<true, false, 2, T, T, T>(l_y, l_x1, l_x2_, l_x3, output_dims, output_stride, mask_stride,
                                             input_stride);
      } else if (param.x3_bcast) {
        WhereExtend<false, true, 2, T, T, T>(l_y, l_x1, l_x2, l_x3_, output_dims, output_stride, mask_stride,
                                             input_stride);
      } else {
        WhereExtend<false, false, 2, T, T, T>(l_y, l_x1, l_x2, l_x3, output_dims, output_stride, mask_stride,
                                              input_stride);
      }
    }

    UbToGm(param.y, l_y, param.y_stride * param.m);
  }
  template <typename T, uint8_t dim>
  static void InvokeNormalKernelInt64(WhereInputParam<T, dim> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, x3buf, ybuf;
    tpipe.InitBuffer(x1buf, sizeof(uint8_t) * param.x1_stride * param.m);
    tpipe.InitBuffer(x2buf, param.x2_bcast ? ONE_BLK_SIZE : sizeof(int64_t) * param.x2_stride * param.m);
    tpipe.InitBuffer(x3buf, param.x3_bcast ? ONE_BLK_SIZE : sizeof(int64_t) * param.x3_stride * param.m);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * param.y_stride * param.m);

    LocalTensor<uint8_t> l_x1 = x1buf.Get<uint8_t>();
    LocalTensor<int64_t> l_x2 = x2buf.Get<int64_t>();
    LocalTensor<int64_t> l_x3 = x3buf.Get<int64_t>();
    LocalTensor<int64_t> l_y = ybuf.Get<int64_t>();

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

    if constexpr (dim == 1) {
      const uint16_t output_dims[dim] = {param.size};
      const uint16_t output_stride[dim] = {1};
      const uint16_t mask_stride[dim] = {1};
      const uint16_t input_stride[dim] = {1};

      if (param.x2_bcast && param.x3_bcast) {
        WhereExtend<true, true, 1, T, T, T>(l_y, l_x1, l_x2, l_x3, output_dims, output_stride, mask_stride,
                                            input_stride);
      } else if (param.x2_bcast) {
        WhereExtend<true, false, 1, T, T, T>(l_y, l_x1, l_x2, l_x3, output_dims, output_stride, mask_stride,
                                             input_stride);
      } else if (param.x3_bcast) {
        WhereExtend<false, true, 1, T, T, T>(l_y, l_x1, l_x2, l_x3, output_dims, output_stride, mask_stride,
                                             input_stride);
      } else {
        WhereExtend<false, false, 1, T, T, T>(l_y, l_x1, l_x2, l_x3, output_dims, output_stride, mask_stride,
                                              input_stride);
      }
    } else if constexpr (dim == 2) {
      const uint16_t output_dims[dim] = {param.m, param.size};
      const uint16_t output_stride[dim] = {param.y_stride, 1};
      const uint16_t mask_stride[dim] = {param.x1_stride, 1};
      const uint16_t input_stride[dim] = {param.x2_stride, 1};

      if (param.x2_bcast && param.x3_bcast) {
        WhereExtend<true, true, 2, T, T, T>(l_y, l_x1, l_x2, l_x3, output_dims, output_stride, mask_stride,
                                            input_stride);
      } else if (param.x2_bcast) {
        WhereExtend<true, false, 2, T, T, T>(l_y, l_x1, l_x2, l_x3, output_dims, output_stride, mask_stride,
                                             input_stride);
      } else if (param.x3_bcast) {
        WhereExtend<false, true, 2, T, T, T>(l_y, l_x1, l_x2, l_x3, output_dims, output_stride, mask_stride,
                                             input_stride);
      } else {
        WhereExtend<false, false, 2, T, T, T>(l_y, l_x1, l_x2, l_x3, output_dims, output_stride, mask_stride,
                                              input_stride);
      }
    }

    UbToGm(param.y, l_y, param.y_stride * param.m);
  }

  template <typename T, uint8_t dim>
  static void FreeNormalInput(WhereInputParam<T, dim> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.x1);
    AscendC::GmFree(param.x2);
    AscendC::GmFree(param.x3);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.x2_);
    AscendC::GmFree(param.x3_);
  }

  template <typename T, uint8_t dim>
  static void FreeNormalInputInt64(WhereInputParam<T, dim> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.x1);
    AscendC::GmFree(param.x2);
    AscendC::GmFree(param.x3);
    AscendC::GmFree(param.exp);
  }

  template <typename T, uint8_t dim>
  static void WhereNormalTest(uint32_t m, uint32_t n, uint32_t x2_n = 0, uint32_t x3_n = 0) {
    WhereInputParam<T, dim> param{};
    param.m = m;
    param.size = n;
    param.x2_size = x2_n == 0 ? n : ONE_BLK_SIZE / sizeof(T);
    param.x2_bcast = x2_n == 0 ? false : true;
    param.x3_size = x3_n == 0 ? n : ONE_BLK_SIZE / sizeof(T);
    param.x3_bcast = x3_n == 0 ? false : true;

    CreateNormalInput(param);

    // 构造Api调用函数
    auto kernel = [&param]() { InvokeNormalKernel<T>(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = NormalValid(param);
    EXPECT_EQ(diff_count, 0);

    // 释放内存
    FreeNormalInput(param);
  }

  template <typename T, uint8_t dim>
  static void WhereNormalTestInt64(uint32_t m, uint32_t n, uint32_t x2_n = 0, uint32_t x3_n = 0) {
    WhereInputParam<T, dim> param{};
    param.m = m;
    param.size = n;
    param.x2_size = x2_n == 0 ? n : ONE_BLK_SIZE / sizeof(int64_t);
    param.x2_bcast = x2_n == 0 ? false : true;
    param.x3_size = x3_n == 0 ? n : ONE_BLK_SIZE / sizeof(int64_t);
    param.x3_bcast = x3_n == 0 ? false : true;

    CreateNormalInputInt64(param);

    // 构造Api调用函数
    auto kernel = [&param]() { InvokeNormalKernelInt64(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = NormalValid(param);
    EXPECT_EQ(diff_count, 0);

    // 释放内存
    FreeNormalInputInt64(param);
  }
};
// 场景1
TEST_F(TestRegbaseApiWhereUT, Where_X2S_X3S_int64_count) {
  WhereNormalTestInt64<int64_t, 1>(1, ONE_BLK_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTestInt64<int64_t, 1>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTestInt64<int64_t, 1>(1, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 1, 1);
  WhereNormalTestInt64<int64_t, 1>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 1, 1);
}

TEST_F(TestRegbaseApiWhereUT, Where_X2S_X3S_int64_normal) {
  WhereNormalTestInt64<int64_t, 2>(3, ONE_BLK_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTestInt64<int64_t, 2>(4, ONE_REPEAT_BYTE_SIZE / sizeof(int64_t), 1, 1);
  WhereNormalTestInt64<int64_t, 2>(5, (ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t), 1, 1);
  WhereNormalTestInt64<int64_t, 2>(18, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t), 1, 1);
}

// 场景2
TEST_F(TestRegbaseApiWhereUT, Where_X2S_float_count) {
  WhereNormalTest<float, 1>(1, ONE_BLK_SIZE / sizeof(float), 1, 0);
  WhereNormalTest<float, 1>(1, ONE_REPEAT_BYTE_SIZE / sizeof(float), 1, 0);
  WhereNormalTest<float, 1>(1, (ONE_BLK_SIZE - sizeof(float)) / sizeof(float), 1, 0);
  WhereNormalTest<float, 1>(1, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float), 1, 0);
}
// 场景3
TEST_F(TestRegbaseApiWhereUT, Where_X3S_int32_count) {
  WhereNormalTest<int32_t, 1>(1, ONE_BLK_SIZE / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t, 1>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t, 1>(1, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 0, 1);
}

TEST_F(TestRegbaseApiWhereUT, Where_X3S_int32_normal) {
  WhereNormalTest<int32_t, 2>(71, ONE_BLK_SIZE / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t, 2>(71, ONE_REPEAT_BYTE_SIZE / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t, 2>(71, (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 0, 1);
  WhereNormalTest<int32_t, 2>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t), 0, 1);
}

// 场景4
TEST_F(TestRegbaseApiWhereUT, Where_int16_count) {
  WhereNormalTest<int16_t, 1>(1, ONE_BLK_SIZE / sizeof(int16_t));
  WhereNormalTest<int16_t, 1>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  WhereNormalTest<int16_t, 1>(1, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
}

TEST_F(TestRegbaseApiWhereUT, Where_int16_normal) {
  WhereNormalTest<int16_t, 2>(71, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  WhereNormalTest<int16_t, 2>(71, (ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
  WhereNormalTest<int16_t, 2>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t));
}

// 场景5: bf16 测试
TEST_F(TestRegbaseApiWhereUT, Where_bf16_count) {
  WhereNormalTest<bfloat16_t, 1>(1, ONE_BLK_SIZE / sizeof(bfloat16_t));
  WhereNormalTest<bfloat16_t, 1>(1, ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  WhereNormalTest<bfloat16_t, 1>(1, (ONE_BLK_SIZE - sizeof(bfloat16_t)) / sizeof(bfloat16_t));
}

TEST_F(TestRegbaseApiWhereUT, Where_bf16_normal) {
  WhereNormalTest<bfloat16_t, 2>(71, ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  WhereNormalTest<bfloat16_t, 2>(71, (ONE_BLK_SIZE - sizeof(bfloat16_t)) / sizeof(bfloat16_t));
  WhereNormalTest<bfloat16_t, 2>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(bfloat16_t));
}

// 场景6: int8 测试
TEST_F(TestRegbaseApiWhereUT, Where_int8_count) {
  WhereNormalTest<int8_t, 1>(1, ONE_BLK_SIZE / sizeof(int8_t));
  WhereNormalTest<int8_t, 1>(1, ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  WhereNormalTest<int8_t, 1>(1, (ONE_BLK_SIZE - sizeof(int8_t)) / sizeof(int8_t));
}

TEST_F(TestRegbaseApiWhereUT, Where_int8_normal) {
  WhereNormalTest<int8_t, 2>(71, ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  WhereNormalTest<int8_t, 2>(71, (ONE_BLK_SIZE - sizeof(int8_t)) / sizeof(int8_t));
  WhereNormalTest<int8_t, 2>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int8_t));
}

// 场景7: uint8 测试
TEST_F(TestRegbaseApiWhereUT, Where_uint8_count) {
  WhereNormalTest<uint8_t, 1>(1, ONE_BLK_SIZE / sizeof(uint8_t));
  WhereNormalTest<uint8_t, 1>(1, ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  WhereNormalTest<uint8_t, 1>(1, (ONE_BLK_SIZE - sizeof(uint8_t)) / sizeof(uint8_t));
}

TEST_F(TestRegbaseApiWhereUT, Where_uint8_normal) {
  WhereNormalTest<uint8_t, 2>(71, ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  WhereNormalTest<uint8_t, 2>(71, (ONE_BLK_SIZE - sizeof(uint8_t)) / sizeof(uint8_t));
  WhereNormalTest<uint8_t, 2>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint8_t));
}

// 场景8: uint16 测试
TEST_F(TestRegbaseApiWhereUT, Where_uint16_count) {
  WhereNormalTest<uint16_t, 1>(1, ONE_BLK_SIZE / sizeof(uint16_t));
  WhereNormalTest<uint16_t, 1>(1, ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t));
  WhereNormalTest<uint16_t, 1>(1, (ONE_BLK_SIZE - sizeof(uint16_t)) / sizeof(uint16_t));
}

TEST_F(TestRegbaseApiWhereUT, Where_uint16_normal) {
  WhereNormalTest<uint16_t, 2>(71, ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t));
  WhereNormalTest<uint16_t, 2>(71, (ONE_BLK_SIZE - sizeof(uint16_t)) / sizeof(uint16_t));
  WhereNormalTest<uint16_t, 2>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint16_t));
}

// 场景9: uint32 测试
TEST_F(TestRegbaseApiWhereUT, Where_uint32_count) {
  WhereNormalTest<uint32_t, 1>(1, ONE_BLK_SIZE / sizeof(uint32_t));
  WhereNormalTest<uint32_t, 1>(1, ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
  WhereNormalTest<uint32_t, 1>(1, (ONE_BLK_SIZE - sizeof(uint32_t)) / sizeof(uint32_t));
}

TEST_F(TestRegbaseApiWhereUT, Where_uint32_normal) {
  WhereNormalTest<uint32_t, 2>(71, ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
  WhereNormalTest<uint32_t, 2>(71, (ONE_BLK_SIZE - sizeof(uint32_t)) / sizeof(uint32_t));
  WhereNormalTest<uint32_t, 2>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint32_t));
}

// 场景10: uint64 测试
TEST_F(TestRegbaseApiWhereUT, Where_uint64_count) {
  WhereNormalTest<uint64_t, 1>(1, ONE_BLK_SIZE / sizeof(uint64_t));
  WhereNormalTest<uint64_t, 1>(1, ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t));
  WhereNormalTest<uint64_t, 1>(1, (ONE_BLK_SIZE - sizeof(uint64_t)) / sizeof(uint64_t));
}

TEST_F(TestRegbaseApiWhereUT, Where_uint64_normal) {
  WhereNormalTest<uint64_t, 2>(71, ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t));
  WhereNormalTest<uint64_t, 2>(71, (ONE_BLK_SIZE - sizeof(uint64_t)) / sizeof(uint64_t));
  WhereNormalTest<uint64_t, 2>(71, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint64_t));
}

// 场景10b: uint64 双标量广播（场景4）
TEST_F(TestRegbaseApiWhereUT, Where_X2S_X3S_uint64_count) {
  WhereNormalTest<uint64_t, 1>(1, ONE_BLK_SIZE / sizeof(uint64_t), 1, 1);
  WhereNormalTest<uint64_t, 1>(1, ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t), 1, 1);
  WhereNormalTest<uint64_t, 1>(1, (ONE_BLK_SIZE - sizeof(uint64_t)) / sizeof(uint64_t), 1, 1);
}

TEST_F(TestRegbaseApiWhereUT, Where_X2S_X3S_uint64_normal) {
  WhereNormalTest<uint64_t, 2>(3, ONE_BLK_SIZE / sizeof(uint64_t), 1, 1);
  WhereNormalTest<uint64_t, 2>(4, ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t), 1, 1);
  WhereNormalTest<uint64_t, 2>(18, (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint64_t), 1, 1);
}

}  // namespace af
