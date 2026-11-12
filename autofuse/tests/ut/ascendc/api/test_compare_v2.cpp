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

TEST(TestApiCompareV2Regression, ScalarBlockInLargerAllocation) {
  constexpr uint32_t rows = 260;
  constexpr uint32_t input_stride = ONE_BLK_SIZE / sizeof(int32_t);
  constexpr uint32_t output_stride = ONE_BLK_SIZE;
  auto *input = static_cast<int32_t *>(GmAlloc(rows * ONE_BLK_SIZE));
  auto *threshold = static_cast<int32_t *>(GmAlloc(rows * ONE_BLK_SIZE));
  auto *output = static_cast<uint8_t *>(GmAlloc(rows * output_stride));
  for (uint32_t i = 0; i < rows * input_stride; ++i) {
    input[i] = static_cast<int32_t>((i / input_stride) % 7) - 3;
    threshold[i] = i < input_stride ? 2 : -99;
  }
  auto kernel = [](int32_t *input, int32_t *threshold, uint8_t *output) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> xbuf, scalarbuf, ybuf, tmpbuf;
    pipe.InitBuffer(xbuf, rows * ONE_BLK_SIZE);
    pipe.InitBuffer(scalarbuf, rows * ONE_BLK_SIZE);
    pipe.InitBuffer(ybuf, rows * output_stride);
    pipe.InitBuffer(tmpbuf, 64 * 1024);
    auto x = xbuf.Get<int32_t>();
    auto scalar = scalarbuf.Get<int32_t>();
    auto y = ybuf.Get<uint8_t>();
    auto tmp = tmpbuf.Get<uint8_t>();
    GmToUb(x, input, rows * input_stride);
    GmToUb(scalar, threshold, rows * input_stride);
    // Only the first block is valid, despite sharing a much larger allocation.
    CompareExtend<int32_t, CMPMODE::LT>(y, x, scalar, rows, 1, input_stride, output_stride, tmp, ONE_BLK_SIZE);
    UbToGm(output, y, rows * output_stride);
  };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, input, threshold, output);
  uint32_t mismatches = 0;
  for (uint32_t row = 0; row < rows; ++row) {
    mismatches += output[row * output_stride] != static_cast<uint8_t>(input[row * input_stride] < 2);
  }
  EXPECT_EQ(mismatches, 0U);
  GmFree(input);
  GmFree(threshold);
  GmFree(output);
}

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

}  // namespace af
