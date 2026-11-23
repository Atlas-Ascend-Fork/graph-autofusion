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
 * test_compare_v2.cpp
 */

#include <cmath>
#include "gtest/gtest.h"
#include "test_api_utils.h"
#include "tikicpulib.h"
#include "utils.h"
// 保持在utils.h之后
#include "duplicate.h"
// 保持在duplicate.h之后
#include "compare_v2.h"

using namespace AscendC;

namespace af {
template <typename O, typename I>
struct TensorCompareV2InputParam {
  O *y{};
  bool *exp{};
  I *src0{};
  I *src1{};
  uint32_t size{0};
  uint32_t out_size{0};
  BinaryRepeatParams a;
};

class TestApiCompareV2UT : public testing::Test {
 protected:
  template <CMPMODE mode, typename O, typename I>
  static void InvokeKernelWithTwoTensorInput(TensorCompareV2InputParam<O, I> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;

    tpipe.InitBuffer(x1buf, sizeof(I) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(I) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(O) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(O)));
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<I> l_x1 = x1buf.Get<I>();
    LocalTensor<I> l_x2 = x2buf.Get<I>();

    LocalTensor<O> l_y = ybuf.Get<O>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, param.size);
    GmToUb(l_x2, param.src1, param.size);
    CompareExtend<I, mode>(l_y, l_x1, l_x2, 1, param.size, (param.size * sizeof(I) / ONE_BLK_SIZE + 1) * ONE_BLK_SIZE,
                           (param.size * sizeof(O) / ONE_BLK_SIZE + 1) * ONE_BLK_SIZE, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <CMPMODE mode, typename O, typename I>
  static void CreateTensorInput(TensorCompareV2InputParam<O, I> &param, float def_src1) {
    // 构造测试输入和预期结果
    param.y = static_cast<O *>(AscendC::GmAlloc(sizeof(O) * param.size));
    param.exp = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * param.size));
    param.src0 = static_cast<I *>(AscendC::GmAlloc(sizeof(I) * param.size));
    param.src1 = static_cast<I *>(AscendC::GmAlloc(sizeof(I) * param.size));
    I src1_val;
    if constexpr (std::is_same_v<I, int64_t>) {
      src1_val = 0xAAAAAAAABBBBBBBB;
    } else {
      src1_val = def_src1;
    }
    (void)src1_val;

    int input_range = 10;

    // 构造src0的随机生成器
    std::mt19937 eng(1);
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    // 构造src1的随机生成器
    std::mt19937 eng1(3);                                  // Seed the generator
    std::uniform_int_distribution distr1(0, input_range);  // Define the range

    for (int i = 0; i < param.size; i++) {
      auto input = distr(eng);  // Use the secure random number generator
      auto input1 = distr1(eng1);
      param.src0[i] = input;
      param.src1[i] = input1;

      switch (mode) {
        case CMPMODE::EQ:
          if (input > 5 || i == param.size - 1) {
            param.src0[i] = param.src1[i];
            param.exp[i] = true;
          } else {
            param.exp[i] = DefaultCompare(param.src0[i], param.src1[i]);
          }
          break;
        case CMPMODE::NE:
          if (input > 5 || i == param.size - 1) {
            param.src0[i] = param.src1[i];
            param.exp[i] = false;
          } else {
            param.exp[i] = !DefaultCompare(param.src0[i], param.src1[i]);
          }
          break;
        case CMPMODE::GE:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(param.src0[i]) >= param.src1[i];
          } else {
            param.exp[i] = param.src0[i] >= param.src1[i];
          }
          break;
        case CMPMODE::LE:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(param.src0[i]) <= param.src1[i];
          } else {
            param.exp[i] = param.src0[i] <= param.src1[i];
          }
          break;
        case CMPMODE::GT:
          param.exp[i] = param.src0[i] > param.src1[i];
          break;
        case CMPMODE::LT:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(param.src0[i]) < param.src1[i];
          } else {
            param.exp[i] = param.src0[i] < param.src1[i];
          }
          break;
        default:
          break;
      }
    }
  }

  template <typename O>
  static uint32_t Valid(O *y, bool *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (static_cast<bool>(y[i]) != exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <CMPMODE mode, typename O, typename I>
  static void TensorCompareV2Test(uint32_t size, float def_src1 = 4.5) {
    TensorCompareV2InputParam<O, I> param{};
    param.size = size;

    CreateTensorInput<mode>(param, def_src1);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernelWithTwoTensorInput<mode>(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0) << "of  " << param.size;
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
    AscendC::GmFree(param.src1);
  }

  template <CMPMODE mode>
  static void TensorCompareV2Int64RegressionTest(uint32_t row_size) {
    constexpr uint32_t rows = 4;
    const uint32_t size = rows * row_size;
    const int64_t src0_values[] = {2, -4, -1, 0, 1, 4, 4294967296LL, -4294967296LL};
    const int64_t src1_values[] = {-8, -5, 0, 0, 2, 4, 4294967295LL, -4294967295LL};
    TensorCompareV2InputParam<uint8_t, int64_t> param{};
    param.size = size;
    param.y = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(uint8_t) * size));
    param.exp = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * size));
    param.src0 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * size));
    param.src1 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * size));
    for (uint32_t i = 0; i < size; i++) {
      const uint32_t index = (i + i / row_size) % 8;
      param.src0[i] = src0_values[index];
      param.src1[i] = src1_values[index];
      param.exp[i] = mode == CMPMODE::GE ? param.src0[i] >= param.src1[i] : param.src0[i] > param.src1[i];
    }

    auto kernel = [&param, row_size] {
      const uint32_t input_stride = AlignUp(row_size, ONE_BLK_SIZE / sizeof(int64_t));
      const uint32_t output_stride = AlignUp(row_size, ONE_BLK_SIZE / sizeof(uint8_t));
      TPipe tpipe;
      TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;
      tpipe.InitBuffer(x1buf, sizeof(int64_t) * rows * input_stride);
      tpipe.InitBuffer(x2buf, sizeof(int64_t) * rows * input_stride);
      tpipe.InitBuffer(ybuf, rows * output_stride);
      tpipe.InitBuffer(tmp, 40 * rows * KernelUtils::Max(row_size, 32U));
      auto lhs = x1buf.Get<int64_t>();
      auto rhs = x2buf.Get<int64_t>();
      auto output = ybuf.Get<uint8_t>();
      auto scratch = tmp.Get<uint8_t>();
      for (uint32_t row = 0; row < rows; ++row) {
        auto lhs_row = lhs[row * input_stride];
        auto rhs_row = rhs[row * input_stride];
        GmToUb(lhs_row, param.src0 + row * row_size, row_size);
        GmToUb(rhs_row, param.src1 + row * row_size, row_size);
      }
      CompareExtend<int64_t, mode>(output, lhs, rhs, rows, row_size, input_stride, output_stride, scratch);
      for (uint32_t row = 0; row < rows; ++row) {
        auto output_row = output[row * output_stride];
        UbToGm(param.y + row * row_size, output_row, row_size);
      }
    };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    EXPECT_EQ(Valid(param.y, param.exp, param.size), 0);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
    AscendC::GmFree(param.src1);
  }

  // 尾块向量化对拍测试：覆盖 last_axis ∈ {4,10,23,24,31}（< 32 走 staging 尾块），
  // 值域含 INT64_MIN/MAX、±(1<<31)、跨 hi 相等/不等、lo 无符号翻转边界（0 vs 0xFFFFFFFF）。
  template <CMPMODE mode>
  static void TensorCompareV2Int64TailVectorTest(uint32_t row_size) {
    constexpr uint32_t rows = 8;
    const uint32_t size = rows * row_size;
    // 覆盖：相等、hi 不等、hi 相等 lo 无符号边界、符号位翻转边界、INT64 极值
    const int64_t src0_values[] = {0, 1, -1, INT64_MAX, INT64_MIN, 4294967295LL, 1LL << 31, -(1LL << 31)};
    const int64_t src1_values[] = {0, 1, 1, INT64_MIN, INT64_MAX, 4294967296LL, -(1LL << 31), 1LL << 31};
    constexpr uint32_t val_cnt = sizeof(src0_values) / sizeof(src0_values[0]);
    TensorCompareV2InputParam<uint8_t, int64_t> param{};
    param.size = size;
    param.y = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(uint8_t) * size));
    param.exp = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * size));
    param.src0 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * size));
    param.src1 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * size));
    for (uint32_t i = 0; i < size; i++) {
      const uint32_t index = i % val_cnt;
      param.src0[i] = src0_values[index];
      param.src1[i] = src1_values[index];
      switch (mode) {
        case CMPMODE::EQ:
          param.exp[i] = param.src0[i] == param.src1[i];
          break;
        case CMPMODE::NE:
          param.exp[i] = param.src0[i] != param.src1[i];
          break;
        case CMPMODE::GT:
          param.exp[i] = param.src0[i] > param.src1[i];
          break;
        case CMPMODE::GE:
          param.exp[i] = param.src0[i] >= param.src1[i];
          break;
        case CMPMODE::LE:
          param.exp[i] = param.src0[i] <= param.src1[i];
          break;
        default:
          param.exp[i] = false;
          break;
      }
    }

    auto kernel = [&param, row_size] {
      const uint32_t input_stride = AlignUp(row_size, ONE_BLK_SIZE / sizeof(int64_t));
      const uint32_t output_stride = AlignUp(row_size, ONE_BLK_SIZE / sizeof(uint8_t));
      TPipe tpipe;
      TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;
      tpipe.InitBuffer(x1buf, sizeof(int64_t) * rows * input_stride);
      tpipe.InitBuffer(x2buf, sizeof(int64_t) * rows * input_stride);
      tpipe.InitBuffer(ybuf, rows * output_stride);
      // 尾块 staging 实现的 tmp 需求与行数无关（Int64CompareTailRowBytes ≈ 2.5KB+），给足裕量
      tpipe.InitBuffer(tmp, 32 * 1024);
      auto lhs = x1buf.Get<int64_t>();
      auto rhs = x2buf.Get<int64_t>();
      auto output = ybuf.Get<uint8_t>();
      auto scratch = tmp.Get<uint8_t>();
      for (uint32_t row = 0; row < rows; ++row) {
        auto lhs_row = lhs[row * input_stride];
        auto rhs_row = rhs[row * input_stride];
        GmToUb(lhs_row, param.src0 + row * row_size, row_size);
        GmToUb(rhs_row, param.src1 + row * row_size, row_size);
      }
      CompareExtend<int64_t, mode>(output, lhs, rhs, rows, row_size, input_stride, output_stride, scratch);
      for (uint32_t row = 0; row < rows; ++row) {
        auto output_row = output[row * output_stride];
        UbToGm(param.y + row * row_size, output_row, row_size);
      }
    };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    EXPECT_EQ(Valid(param.y, param.exp, param.size), 0);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
    AscendC::GmFree(param.src1);
  }

  template <CMPMODE mode, typename O, typename I>
  static void InvokeKernelWithBroadcastTensorInput(TensorCompareV2InputParam<O, I> &param, uint32_t first_axis,
                                                   uint32_t row_size, uint32_t out_stride) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;

    tpipe.InitBuffer(x1buf, sizeof(I) * first_axis * row_size);
    tpipe.InitBuffer(x2buf, ONE_BLK_SIZE);
    tpipe.InitBuffer(ybuf, sizeof(O) * first_axis * out_stride);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<I> l_x1 = x1buf.Get<I>();
    LocalTensor<I> l_x2 = x2buf.Get<I>();
    LocalTensor<O> l_y = ybuf.Get<O>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, first_axis * row_size);
    GmToUb(l_x2, param.src1, row_size);
    CompareExtend<I, mode>(l_y, l_x1, l_x2, first_axis, row_size, row_size, out_stride, l_tmp);
    for (uint32_t i = 0; i < first_axis; i++) {
      for (uint32_t j = 0; j < row_size; j++) {
        param.y[i * row_size + j] = l_y.GetValue(i * out_stride + j);
      }
    }
  }

  template <CMPMODE mode, typename O, typename I>
  static void CreateBroadcastTensorInput(TensorCompareV2InputParam<O, I> &param, uint32_t first_axis,
                                         uint32_t row_size) {
    const uint32_t total_size = first_axis * row_size;
    // 构造测试输入和预期结果：src0为(first_axis, row_size)，src1为1行(row_size)，按行广播
    param.y = static_cast<O *>(AscendC::GmAlloc(sizeof(O) * total_size));
    param.exp = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * total_size));
    param.src0 = static_cast<I *>(AscendC::GmAlloc(sizeof(I) * total_size));
    param.src1 = static_cast<I *>(AscendC::GmAlloc(sizeof(I) * row_size));

    int input_range = 10;
    std::mt19937 eng(1);
    std::uniform_int_distribution distr(0, input_range);
    std::mt19937 eng1(3);
    std::uniform_int_distribution distr1(0, input_range);

    for (uint32_t j = 0; j < row_size; j++) {
      param.src1[j] = distr1(eng1);
    }
    for (uint32_t i = 0; i < total_size; i++) {
      param.src0[i] = distr(eng);
    }
    for (uint32_t i = 0; i < first_axis; i++) {
      for (uint32_t j = 0; j < row_size; j++) {
        const I lhs = param.src0[i * row_size + j];
        const I rhs = param.src1[j];
        switch (mode) {
          case CMPMODE::EQ:
            param.exp[i * row_size + j] = lhs == rhs;
            break;
          case CMPMODE::NE:
            param.exp[i * row_size + j] = lhs != rhs;
            break;
          case CMPMODE::GE:
            if constexpr (std::is_same_v<I, half>) {
              param.exp[i * row_size + j] = static_cast<half>(lhs) >= rhs;
            } else {
              param.exp[i * row_size + j] = lhs >= rhs;
            }
            break;
          case CMPMODE::LE:
            if constexpr (std::is_same_v<I, half>) {
              param.exp[i * row_size + j] = static_cast<half>(lhs) <= rhs;
            } else {
              param.exp[i * row_size + j] = lhs <= rhs;
            }
            break;
          case CMPMODE::GT:
            param.exp[i * row_size + j] = lhs > rhs;
            break;
          case CMPMODE::LT:
            if constexpr (std::is_same_v<I, half>) {
              param.exp[i * row_size + j] = static_cast<half>(lhs) < rhs;
            } else {
              param.exp[i * row_size + j] = lhs < rhs;
            }
            break;
          default:
            break;
        }
      }
    }
  }

  template <CMPMODE mode, typename O, typename I>
  static void TensorCompareV2BroadcastTest(uint32_t first_axis) {
    TensorCompareV2InputParam<O, I> param{};
    // 广播行恰占一个block，满足CompareExtend的32B广播判定契约
    const uint32_t row_size = ONE_BLK_SIZE / sizeof(I);
    const uint32_t out_stride = AlignUp(row_size, ONE_BLK_SIZE / sizeof(O));

    CreateBroadcastTensorInput<mode>(param, first_axis, row_size);

    // 构造Api调用函数
    auto kernel = [&param, first_axis, row_size, out_stride] {
      InvokeKernelWithBroadcastTensorInput<mode>(param, first_axis, row_size, out_stride);
    };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.exp, first_axis * row_size);
    EXPECT_EQ(diff_count, 0) << "of first_axis " << first_axis;
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
    AscendC::GmFree(param.src1);
  }
};

// 场景1：GT-int32
TEST_F(TestApiCompareV2UT, CompareV2_Gt_input_tensor_int32_output_uint8) {
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) /
                                                     sizeof(int32_t));
}

// 场景2：NE-int32
TEST_F(TestApiCompareV2UT, CompareV2_Ne_input_tensor_int32_output_uint8) {
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::NE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) /
                                                     sizeof(int32_t));
}

// 场景3：LE-int32
TEST_F(TestApiCompareV2UT, CompareV2_Le_input_tensor_int32_output_uint8) {
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) /
                                                     sizeof(int32_t));
}

// 场景4：GE-int32
TEST_F(TestApiCompareV2UT, CompareV2_Ge_input_tensor_int32_output_uint8) {
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::GE, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) /
                                                     sizeof(int32_t));
}

// 场景5.1：GE-int64，覆盖多repeat的正数、负数及相等值。
TEST_F(TestApiCompareV2UT, CompareV2_Ge_input_tensor_int64_output_uint8) {
  TensorCompareV2Int64RegressionTest<CMPMODE::GE>(4);
  TensorCompareV2Int64RegressionTest<CMPMODE::GE>(16);
  TensorCompareV2Int64RegressionTest<CMPMODE::GE>(32);
  TensorCompareV2Int64RegressionTest<CMPMODE::GE>(40);
  TensorCompareV2Int64RegressionTest<CMPMODE::GE>(100);
  TensorCompareV2Int64RegressionTest<CMPMODE::GE>(128);
}

// 场景5.2：GT-int64，覆盖多repeat的正数、负数及相等值。
TEST_F(TestApiCompareV2UT, CompareV2_Gt_input_tensor_int64_output_uint8) {
  TensorCompareV2Int64RegressionTest<CMPMODE::GT>(4);
  TensorCompareV2Int64RegressionTest<CMPMODE::GT>(16);
  TensorCompareV2Int64RegressionTest<CMPMODE::GT>(32);
  TensorCompareV2Int64RegressionTest<CMPMODE::GT>(40);
  TensorCompareV2Int64RegressionTest<CMPMODE::GT>(100);
  TensorCompareV2Int64RegressionTest<CMPMODE::GT>(128);
}

// 场景5.3：EQ-int64 尾块（staging），last_axis ∈ {4,10,23,24,31}
TEST_F(TestApiCompareV2UT, CompareV2_Eq_input_tensor_int64_tail_vector) {
  TensorCompareV2Int64TailVectorTest<CMPMODE::EQ>(4);
  TensorCompareV2Int64TailVectorTest<CMPMODE::EQ>(10);
  TensorCompareV2Int64TailVectorTest<CMPMODE::EQ>(23);
  TensorCompareV2Int64TailVectorTest<CMPMODE::EQ>(24);
  TensorCompareV2Int64TailVectorTest<CMPMODE::EQ>(31);
}

// 场景5.4：NE-int64 尾块（staging）
TEST_F(TestApiCompareV2UT, CompareV2_Ne_input_tensor_int64_tail_vector) {
  TensorCompareV2Int64TailVectorTest<CMPMODE::NE>(4);
  TensorCompareV2Int64TailVectorTest<CMPMODE::NE>(10);
  TensorCompareV2Int64TailVectorTest<CMPMODE::NE>(23);
  TensorCompareV2Int64TailVectorTest<CMPMODE::NE>(24);
  TensorCompareV2Int64TailVectorTest<CMPMODE::NE>(31);
}

// 场景5.5：GT-int64 尾块（staging）
TEST_F(TestApiCompareV2UT, CompareV2_Gt_input_tensor_int64_tail_vector) {
  TensorCompareV2Int64TailVectorTest<CMPMODE::GT>(4);
  TensorCompareV2Int64TailVectorTest<CMPMODE::GT>(10);
  TensorCompareV2Int64TailVectorTest<CMPMODE::GT>(23);
  TensorCompareV2Int64TailVectorTest<CMPMODE::GT>(24);
  TensorCompareV2Int64TailVectorTest<CMPMODE::GT>(31);
}

// 场景5.6：GE-int64 尾块（staging）
TEST_F(TestApiCompareV2UT, CompareV2_Ge_input_tensor_int64_tail_vector) {
  TensorCompareV2Int64TailVectorTest<CMPMODE::GE>(4);
  TensorCompareV2Int64TailVectorTest<CMPMODE::GE>(10);
  TensorCompareV2Int64TailVectorTest<CMPMODE::GE>(23);
  TensorCompareV2Int64TailVectorTest<CMPMODE::GE>(24);
  TensorCompareV2Int64TailVectorTest<CMPMODE::GE>(31);
}

// 场景5.7：LE-int64 尾块（staging）
TEST_F(TestApiCompareV2UT, CompareV2_Le_input_tensor_int64_tail_vector) {
  TensorCompareV2Int64TailVectorTest<CMPMODE::LE>(4);
  TensorCompareV2Int64TailVectorTest<CMPMODE::LE>(10);
  TensorCompareV2Int64TailVectorTest<CMPMODE::LE>(23);
  TensorCompareV2Int64TailVectorTest<CMPMODE::LE>(24);
  TensorCompareV2Int64TailVectorTest<CMPMODE::LE>(31);
}

// 场景5：LT-int32
TEST_F(TestApiCompareV2UT, CompareV2_Lt_input_tensor_int32_output_uint8) {
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>((ONE_BLK_SIZE + sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + sizeof(int32_t)) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE) / sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE - sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE + sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(int32_t)) /
                                                     sizeof(int32_t));
  TensorCompareV2Test<CMPMODE::LT, uint8_t, int32_t>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int32_t)) /
                                                     sizeof(int32_t));
}

/* End: 输入是两个tensor的 compare 测试--------------------------------------- */

/* Begin: 输入是tensor广播单block tensor的 compare 测试------------------------ */

// 场景6：GT-int32广播（src1为单block张量，覆盖CompareExtendInt32的广播分支）。
// 大repeat_times受ProcessSmallAxisCase临时空间约束、CompareNormalNoLoop通用广播分支
// 在first_axis>1时存在已知问题，待compare_v2.h侧修复后补充覆盖。
TEST_F(TestApiCompareV2UT, CompareV2_Gt_input_broadcast_int32_output_uint8) {
  TensorCompareV2BroadcastTest<CMPMODE::GT, uint8_t, int32_t>(2);
}

// 场景7：NE-half广播（覆盖NaN修正路径的==32判定，无NaN输入下结果应与非广播语义一致）
TEST_F(TestApiCompareV2UT, CompareV2_Ne_input_broadcast_half_output_uint8) {
  TensorCompareV2BroadcastTest<CMPMODE::NE, uint8_t, half>(2);
  TensorCompareV2BroadcastTest<CMPMODE::NE, uint8_t, half>(5);
}

/* End: 输入是tensor广播单block tensor的 compare 测试-------------------------- */

/* Begin: ST 门禁（5.2 节）—— Compare+Select 满 repeat、PairReduceSum 打包语义 ---- */

// ST-1：Compare(EQ)+Select(TENSOR_TENSOR) 在 64 lane 满 repeat 下的逐 lane 语义。
// 注：ICPU 上 int32 的 vcmpv 特化只支持 EQ（dav_c220），非 EQ 比较由整块 int64 路径的
// sign/value bits 组合实现，此处仅钉死 EQ 语义。
TEST_F(TestApiCompareV2UT, ST1_CompareSelectBitPredicate) {
  constexpr uint32_t kLanes = 64;  // 256B / 4B
  auto *a = static_cast<int32_t *>(AscendC::GmAlloc(sizeof(int32_t) * kLanes));
  auto *b = static_cast<int32_t *>(AscendC::GmAlloc(sizeof(int32_t) * kLanes));
  auto *out = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kLanes));
  auto *bits_out = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(uint8_t) * 16));
  for (uint32_t i = 0; i < kLanes; i++) {
    a[i] = static_cast<int32_t>(i);
    b[i] = static_cast<int32_t>(32);  // EQ(i, 32)：仅 lane 32 为真
  }
  auto kernel = [a, b, out, bits_out] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(int32_t) * kLanes);
    tpipe.InitBuffer(x2buf, sizeof(int32_t) * kLanes);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);
    auto la = x1buf.Get<int32_t>();
    auto lb = x2buf.Get<int32_t>();
    auto scratch = tmp.Get<uint8_t>();
    GmToUb(la, a, kLanes);
    GmToUb(lb, b, kLanes);
    LocalTensor<uint8_t> bits = scratch[0].template ReinterpretCast<uint8_t>();
    bits.SetSize(kLanes / 8);
    LocalTensor<float> ones = scratch[32].template ReinterpretCast<float>();
    ones.SetSize(kLanes);
    LocalTensor<float> zeros = scratch[32 + ONE_REPEAT_BYTE_SIZE].template ReinterpretCast<float>();
    zeros.SetSize(kLanes);
    LocalTensor<float> res = scratch[32 + 2 * ONE_REPEAT_BYTE_SIZE].template ReinterpretCast<float>();
    res.SetSize(kLanes);
    Duplicate(ones, 1.0f, kLanes);
    Duplicate(zeros, 0.0f, kLanes);
    const BinaryRepeatParams rpt = {1, 1, 1, 8, 8, 8};
    Compare(bits, la, lb, CMPMODE::EQ, kLanes, 1, rpt);
    UbToGm(bits_out, bits, 8);
    Select<float, uint8_t, true>(res, bits, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, kLanes, 1, rpt);
    UbToGm(out, res, kLanes);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  uint32_t diff = 0;
  for (uint32_t i = 0; i < kLanes; i++) {
    bool exp = static_cast<int32_t>(i) == 32;
    if ((exp && out[i] != 1.0f) || (!exp && out[i] != 0.0f)) {
      diff++;
    }
  }
  EXPECT_EQ(diff, 0);
  // packed bits：byte 4 的 bit 0 置位（lane 32）
  EXPECT_EQ(bits_out[4], 0x01);
  AscendC::GmFree(a);
  AscendC::GmFree(b);
  AscendC::GmFree(out);
  AscendC::GmFree(bits_out);
}

// ST-2：PairReduceSum 的相邻 lane 成对求和与紧凑打包布局。
TEST_F(TestApiCompareV2UT, ST2_PairReduceSumPacking) {
  constexpr uint32_t kLanes = 64;  // float lane 数
  constexpr uint32_t kPairs = 32;
  auto *src = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kLanes));
  auto *out = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kPairs));
  for (uint32_t i = 0; i < kLanes; i++) {
    src[i] = static_cast<float>(i);  // [0,1,2,3,...]
  }
  auto kernel = [src, out] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(float) * kLanes);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);
    auto lsrc = x1buf.Get<float>();
    auto scratch = tmp.Get<uint8_t>();
    LocalTensor<float> res = scratch[0].template ReinterpretCast<float>();
    res.SetSize(kPairs);
    GmToUb(lsrc, src, kLanes);
    PairReduceSum(res, lsrc, 1, kLanes, 1, 1, 4);
    UbToGm(out, res, kPairs);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  uint32_t diff = 0;
  for (uint32_t i = 0; i < kPairs; i++) {
    float exp = src[2 * i] + src[2 * i + 1];
    if (out[i] != exp) {
      diff++;
    }
  }
  EXPECT_EQ(diff, 0);
  AscendC::GmFree(src);
  AscendC::GmFree(out);
}

/* End: ST 门禁 ---------------------------------------------------------------- */

}  // namespace af
