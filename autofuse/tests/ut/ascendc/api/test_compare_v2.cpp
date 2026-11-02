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

}  // namespace af
