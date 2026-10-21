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
 * test_compare.cpp
 */

#include <cmath>
#include "gtest/gtest.h"
#include "test_api_utils.h"
#include "tikicpulib.h"
#include "utils.h"
// 保持在utils.h之后
#include "duplicate.h"
// 保持在duplicate.h之后
#include "api_regbase/compare.h"

using namespace AscendC;

namespace af {
template <typename O, typename I, uint8_t dim, CMPMODE mode>
struct CompareInputParam {
  O *y{};
  bool *exp{};
  I *src0{};
  I src1;
  CMPMODE cmpmode{CMPMODE::EQ};
  uint16_t size{0};
  uint16_t out_size{0};
  BinaryRepeatParams a;
};

template <typename O, typename I, uint8_t dim, CMPMODE mode>
struct TensorCompareInputParam {
  O *y{};
  bool *exp{};
  I *src0{};
  I *src1{};
  CMPMODE cmpmode{CMPMODE::EQ};
  uint16_t size{0};
  uint16_t out_size{0};
  uint16_t first_axis{0};
  uint16_t last_axis{0};
  BinaryRepeatParams a;
};

class TestApiCompareUT : public testing::Test {
 protected:
  template <typename O, typename I, uint8_t dim, CMPMODE mode>
  static void CreateInput(CompareInputParam<O, I, dim, mode> &param, float def_src1) {
    // 构造测试输入和预期结果
    param.y = static_cast<O *>(AscendC::GmAlloc(sizeof(O) * param.size));
    param.exp = static_cast<bool *>(AscendC::GmAlloc(sizeof(bool) * param.size));
    param.src0 = static_cast<I *>(AscendC::GmAlloc(sizeof(I) * param.size));

    if constexpr (std::is_same_v<I, int64_t>) {
      param.src1 = 0xAAAAAAAABBBBBBBB;
    } else {
      param.src1 = def_src1;
    }

    int input_range = 10;
    std::mt19937 eng(1);                                  // Seed the generator
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    for (int i = 0; i < param.size; i++) {
      auto input = distr(eng);  // Use the secure random number generator
      param.src0[i] = input;
      // 期望值必须基于截断后实际存储的 src0[i] 计算（如 bool 会把 [0,10] 截断为 0/1），
      // 与内核读取到的数据保持一致
      switch (param.cmpmode) {
        case CMPMODE::EQ:
          if (input > 5 || i == param.size - 1) {
            param.src0[i] = param.src1;
            param.exp[i] = true;
          } else {
            param.exp[i] = DefaultCompare(param.src0[i], param.src1);
          }
          break;
        case CMPMODE::NE:
          if (input > 5 || i == param.size - 1) {
            param.src0[i] = param.src1;
            param.exp[i] = false;
          } else {
            param.exp[i] = !DefaultCompare(param.src0[i], param.src1);
          }
          break;
        case CMPMODE::GE:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(input) >= param.src1;
          } else {
            param.exp[i] = param.src0[i] >= param.src1;
          }
          break;
        case CMPMODE::LE:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(input) <= param.src1;
          } else {
            param.exp[i] = param.src0[i] <= param.src1;
          }
          break;
        case CMPMODE::GT:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(input) > param.src1;
          } else {
            param.exp[i] = param.src0[i] > param.src1;
          }
          break;
        case CMPMODE::LT:
          if constexpr (std::is_same_v<I, half>) {
            param.exp[i] = static_cast<half>(input) < param.src1;
          } else {
            param.exp[i] = param.src0[i] < param.src1;
          }
          break;
        default:
          break;
      }
    }
  }

  template <typename O, typename I, uint8_t dim, CMPMODE mode>
  static uint32_t Valid(CompareInputParam<O, I, dim, mode> &param) {
    uint32_t diff_count = 0;

    for (uint32_t i = 0; i < param.size; i++) {
      if (static_cast<bool>(param.y[i]) != param.exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  // bool 与 uint8 位模式一致：输入数据经 uint8 tensor 装载后以 bool 视图传入被测 API，
  // 规避部分模拟器对 LocalTensor<bool> 装载路径的差异；API 仍按 bool 实例化
  template <typename I>
  static void LoadUbTensor(LocalTensor<I> &local, TBuf<TPosition::VECCALC> &buf, I *gm, int size) {
    if constexpr (std::is_same_v<I, bool>) {
      auto tensorU8 = buf.Get<uint8_t>();
      GmToUb(tensorU8, reinterpret_cast<uint8_t *>(gm), size);
      local = reinterpret_cast<const LocalTensor<I> &>(tensorU8);
    } else {
      local = buf.Get<I>();
      GmToUb(local, gm, size);
    }
  }

  template <typename I>
  static void LoadUbTensorNormal(LocalTensor<I> &local, TBuf<TPosition::VECCALC> &buf, I *gm, uint32_t first_axis,
                                 uint32_t last_axis, uint32_t input_stride) {
    if constexpr (std::is_same_v<I, bool>) {
      auto tensorU8 = buf.Get<uint8_t>();
      GmToUbNormal(tensorU8, reinterpret_cast<uint8_t *>(gm), first_axis, last_axis, input_stride);
      local = reinterpret_cast<const LocalTensor<I> &>(tensorU8);
    } else {
      local = buf.Get<I>();
      GmToUbNormal(local, gm, first_axis, last_axis, input_stride);
    }
  }

  template <typename O, typename I, uint8_t dim, CMPMODE mode>
  static void InvokeKernel(CompareInputParam<O, I, dim, mode> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, ybuf;
    tpipe.InitBuffer(x1buf, sizeof(I) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(O) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(O)));

    LocalTensor<I> l_x1;
    LocalTensor<O> l_y = ybuf.Get<O>();
    LoadUbTensor(l_x1, x1buf, param.src0, param.size);

    const uint16_t output_dims[1] = {param.size};
    const uint16_t output_stride[1] = {1};
    const uint16_t input_stride[1] = {1};
    CompareScalarExtend<I, dim, mode>(l_y, l_x1, param.src1, output_dims, output_stride, input_stride);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename O, typename I, uint8_t dim, CMPMODE mode>
  static void CompareTest(uint16_t size, float def_src1 = 4.5) {
    CompareInputParam<O, I, dim, mode> param{};
    param.size = size;
    param.cmpmode = mode;

    CreateInput(param, def_src1);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
  }

  /* -------------------- 输入是两个tensor相关的测试基础方法定义(count+normal)-------------------- */

  template <typename O, typename I, uint8_t dim, CMPMODE mode>
  static void InvokeKernelWithTwoTensorInput(TensorCompareInputParam<O, I, dim, mode> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf;
    if constexpr (dim == 1) {
      tpipe.InitBuffer(x1buf, sizeof(I) * param.size);
      tpipe.InitBuffer(x2buf, sizeof(I) * param.size);
      tpipe.InitBuffer(ybuf, sizeof(O) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(O)));

      LocalTensor<I> l_x1;
      LocalTensor<I> l_x2;
      LocalTensor<O> l_y = ybuf.Get<O>();

      LoadUbTensor(l_x1, x1buf, param.src0, param.size);
      LoadUbTensor(l_x2, x2buf, param.src1, param.size);
      const uint16_t output_dims[dim] = {param.size};
      const uint16_t output_stride[dim] = {1};
      const uint16_t input_stride[dim] = {1};
      CompareExtend<I, dim, mode>(l_y, l_x1, l_x2, output_dims, output_stride, input_stride);
      UbToGm(param.y, l_y, param.size);
    } else if constexpr (dim == 2) {
      constexpr auto alignInput = ONE_BLK_SIZE / sizeof(I);
      constexpr auto alignOutput = ONE_BLK_SIZE / sizeof(O);
      uint16_t inputStride = CeilDivision(param.last_axis, alignInput) * alignInput;
      uint16_t outputStride = CeilDivision(param.last_axis, alignOutput) * alignOutput;

      uint32_t inputSize = param.first_axis * inputStride;
      uint32_t outputSize = param.first_axis * outputStride;

      tpipe.InitBuffer(x1buf, sizeof(I) * inputSize);
      tpipe.InitBuffer(x2buf, sizeof(I) * inputSize);
      tpipe.InitBuffer(ybuf, sizeof(O) * AlignUp(outputSize, ONE_BLK_SIZE / sizeof(O)));

      LocalTensor<I> l_x1;
      LocalTensor<I> l_x2;
      LocalTensor<O> l_y = ybuf.Get<O>();

      LoadUbTensorNormal(l_x1, x1buf, param.src0, param.first_axis, param.last_axis, inputStride);
      LoadUbTensorNormal(l_x2, x2buf, param.src1, param.first_axis, param.last_axis, inputStride);
      const uint16_t output_dims[dim] = {param.first_axis, param.last_axis};
      const uint16_t output_stride[dim] = {outputStride, 1};
      const uint16_t input_stride[dim] = {inputStride, 1};
      CompareExtend<I, dim, mode>(l_y, l_x1, l_x2, output_dims, output_stride, input_stride);
      UbToGmNormal(param.y, l_y, param.first_axis, param.last_axis, outputStride);
    }
  }

  template <typename O, typename I, uint8_t dim, CMPMODE mode>
  static void CreateTensorInput(TensorCompareInputParam<O, I, dim, mode> &param, float def_src1) {
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
    bool src1IsBlkTensor = false;
    if ((dim == 2) && (param.size * sizeof(I) == ONE_BLK_SIZE)) {
      src1IsBlkTensor = true;
      // 应用场景：外部调用者会把一个标量scalar广播为一个blk大小的tensor作为src1传入compare接口
    }
    for (int i = 0; i < param.size; i++) {
      auto input = distr(eng);  // Use the secure random number generator
      auto input1 = distr1(eng1);
      param.src0[i] = input;
      if (i > 0) {
        if (src1IsBlkTensor) {
          param.src1[i] = param.src1[0];
        } else {
          param.src1[i] = input1;
        }
      } else {
        if constexpr (std::is_same_v<I, bool>) {
          // bool 用例 src1[0] 取 def_src1：blk tensor 广播后 oracle 可控（避免随机值导致期望恒 true）
          param.src1[i] = src1_val;
        } else {
          param.src1[i] = input1;
        }
      }

      switch (param.cmpmode) {
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

  template <typename O, typename I, uint8_t dim, CMPMODE mode>
  static void TensorCompareTestNormal(uint16_t first_axis, uint16_t last_axis, float def_src1 = 4.5) {
    TensorCompareInputParam<O, I, dim, mode> param{};
    param.first_axis = first_axis;
    param.last_axis = last_axis;
    param.size = first_axis * last_axis;
    param.cmpmode = mode;

    CreateTensorInput(param, def_src1);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernelWithTwoTensorInput(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
  template <typename O, typename I, uint8_t dim, CMPMODE mode>
  static void TensorCompareTest(uint16_t size, float def_src1 = 4.5) {
    TensorCompareInputParam<O, I, dim, mode> param{};
    param.size = size;
    param.cmpmode = mode;

    CreateTensorInput(param, def_src1);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernelWithTwoTensorInput(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

// comparescalar的count模式
TEST_F(TestApiCompareUT, CompareScalar_Eq_float_uint8) {
  CompareTest<uint8_t, float, 1, CMPMODE::EQ>(ONE_BLK_SIZE / sizeof(float));
  CompareTest<uint8_t, float, 1, CMPMODE::EQ>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  CompareTest<uint8_t, float, 1, CMPMODE::EQ>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float));
  CompareTest<uint8_t, float, 1, CMPMODE::EQ>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float));
  CompareTest<uint8_t, float, 1, CMPMODE::EQ>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float));
  CompareTest<uint8_t, float, 1, CMPMODE::EQ>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                               (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) /
                                              sizeof(float));
}
// GE
TEST_F(TestApiCompareUT, CompareScalar_Ge_bfloat_uint8) {
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>(ONE_BLK_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) /
                                                   sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                    (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                    (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
                                                   sizeof(bfloat16_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Ge_int8_uint8) {
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GE>(ONE_BLK_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GE>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                (ONE_BLK_SIZE - sizeof(int8_t))) /
                                               sizeof(int8_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Ge_int16_uint8) {
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GE>(ONE_BLK_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GE>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) /
                                                sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                 (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                 (ONE_BLK_SIZE - sizeof(int16_t))) /
                                                sizeof(int16_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Ge_uint8_uint8) {
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>(ONE_BLK_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) /
                                                sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                 (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                 (ONE_BLK_SIZE - sizeof(uint8_t))) /
                                                sizeof(uint8_t));
}
// Gt
TEST_F(TestApiCompareUT, CompareScalar_Gt_bfloat_uint8) {
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>(ONE_BLK_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) /
                                                   sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                    (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                    (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
                                                   sizeof(bfloat16_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Gt_int8_uint8) {
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GT>(ONE_BLK_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GT>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::GT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                (ONE_BLK_SIZE - sizeof(int8_t))) /
                                               sizeof(int8_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Gt_int16_uint8) {
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GT>(ONE_BLK_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GT>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) /
                                                sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::GT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                 (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                 (ONE_BLK_SIZE - sizeof(int16_t))) /
                                                sizeof(int16_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Gt_uint8_uint8) {
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>(ONE_BLK_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) /
                                                sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                 (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                 (ONE_BLK_SIZE - sizeof(uint8_t))) /
                                                sizeof(uint8_t));
}
// Lt
TEST_F(TestApiCompareUT, CompareScalar_Lt_bfloat_uint8) {
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>(ONE_BLK_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) /
                                                   sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                    (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                    (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
                                                   sizeof(bfloat16_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Lt_int8_uint8) {
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LT>(ONE_BLK_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LT>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                (ONE_BLK_SIZE - sizeof(int8_t))) /
                                               sizeof(int8_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Lt_int16_uint8) {
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LT>(ONE_BLK_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LT>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) /
                                                sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                 (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                 (ONE_BLK_SIZE - sizeof(int16_t))) /
                                                sizeof(int16_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Lt_uint8_uint8) {
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LT>(ONE_BLK_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LT>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) /
                                                sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                 (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                 (ONE_BLK_SIZE - sizeof(uint8_t))) /
                                                sizeof(uint8_t));
}
// Le
TEST_F(TestApiCompareUT, CompareScalar_Le_bfloat_uint8) {
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>(ONE_BLK_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) /
                                                   sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  CompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                    (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                    (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
                                                   sizeof(bfloat16_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Le_int8_uint8) {
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LE>(ONE_BLK_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LE>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  CompareTest<uint8_t, int8_t, 1, CMPMODE::LE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                (ONE_BLK_SIZE - sizeof(int8_t))) /
                                               sizeof(int8_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Le_int16_uint8) {
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LE>(ONE_BLK_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LE>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) /
                                                sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  CompareTest<uint8_t, int16_t, 1, CMPMODE::LE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                 (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                 (ONE_BLK_SIZE - sizeof(int16_t))) /
                                                sizeof(int16_t));
}
TEST_F(TestApiCompareUT, CompareScalar_Le_uint8_uint8) {
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>(ONE_BLK_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) /
                                                sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  CompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                 (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                 (ONE_BLK_SIZE - sizeof(uint8_t))) /
                                                sizeof(uint8_t));
}

// compare的count模式
TEST_F(TestApiCompareUT, CompareCount_Eq_float_uint8) {
  TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>(ONE_BLK_SIZE / sizeof(float));
  TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float));
  TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) /
                                                    sizeof(float));
  TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float));
  TensorCompareTest<uint8_t, float, 1, CMPMODE::EQ>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                     (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                     (ONE_BLK_SIZE - sizeof(float))) /
                                                    sizeof(float));
}
TEST_F(TestApiCompareUT, CompareCount_Eq_uint16_uint8) {
  TensorCompareTest<uint8_t, uint16_t, 1, CMPMODE::EQ>(ONE_BLK_SIZE / sizeof(uint16_t));
  TensorCompareTest<uint8_t, uint16_t, 1, CMPMODE::EQ>(ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t));
  TensorCompareTest<uint8_t, uint16_t, 1, CMPMODE::EQ>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t));
  TensorCompareTest<uint8_t, uint16_t, 1, CMPMODE::EQ>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint16_t)) /
                                                       sizeof(uint16_t));
  TensorCompareTest<uint8_t, uint16_t, 1, CMPMODE::EQ>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t));
  TensorCompareTest<uint8_t, uint16_t, 1, CMPMODE::EQ>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                        (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                        (ONE_BLK_SIZE - sizeof(uint16_t))) /
                                                       sizeof(uint16_t));
}
TEST_F(TestApiCompareUT, CompareCount_Eq_uint32_uint8) {
  TensorCompareTest<uint8_t, uint32_t, 1, CMPMODE::EQ>(ONE_BLK_SIZE / sizeof(uint32_t));
  TensorCompareTest<uint8_t, uint32_t, 1, CMPMODE::EQ>(ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
  TensorCompareTest<uint8_t, uint32_t, 1, CMPMODE::EQ>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
  TensorCompareTest<uint8_t, uint32_t, 1, CMPMODE::EQ>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint32_t)) /
                                                       sizeof(uint32_t));
  TensorCompareTest<uint8_t, uint32_t, 1, CMPMODE::EQ>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
  TensorCompareTest<uint8_t, uint32_t, 1, CMPMODE::EQ>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                        (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                        (ONE_BLK_SIZE - sizeof(uint32_t))) /
                                                       sizeof(uint32_t));
}
TEST_F(TestApiCompareUT, CompareCount_Eq_uint64_uint8) {
  TensorCompareTest<uint8_t, uint64_t, 1, CMPMODE::EQ>(ONE_BLK_SIZE / sizeof(uint64_t));
  TensorCompareTest<uint8_t, uint64_t, 1, CMPMODE::EQ>(ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t));
  TensorCompareTest<uint8_t, uint64_t, 1, CMPMODE::EQ>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t));
  TensorCompareTest<uint8_t, uint64_t, 1, CMPMODE::EQ>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint64_t)) /
                                                       sizeof(uint64_t));
  TensorCompareTest<uint8_t, uint64_t, 1, CMPMODE::EQ>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t));
  TensorCompareTest<uint8_t, uint64_t, 1, CMPMODE::EQ>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                        (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                        (ONE_BLK_SIZE - sizeof(uint64_t))) /
                                                       sizeof(uint64_t));
}
// GE
TEST_F(TestApiCompareUT, CompareCount_Ge_bfloat_uint8) {
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>(ONE_BLK_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) /
                                                         sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE /
                                                         sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                          (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                          (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
                                                         sizeof(bfloat16_t));
}
TEST_F(TestApiCompareUT, CompareCount_Ge_int8_uint8) {
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GE>(ONE_BLK_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GE>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) /
                                                     sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                      (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                      (ONE_BLK_SIZE - sizeof(int8_t))) /
                                                     sizeof(int8_t));
}
TEST_F(TestApiCompareUT, CompareCount_Ge_int16_uint8) {
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GE>(ONE_BLK_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GE>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) /
                                                      sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                       (ONE_BLK_SIZE - sizeof(int16_t))) /
                                                      sizeof(int16_t));
}
TEST_F(TestApiCompareUT, CompareCount_Ge_uint8_uint8) {
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>(ONE_BLK_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) /
                                                      sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                       (ONE_BLK_SIZE - sizeof(uint8_t))) /
                                                      sizeof(uint8_t));
}
// Gt
TEST_F(TestApiCompareUT, CompareCount_Gt_bfloat_uint8) {
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>(ONE_BLK_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) /
                                                         sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE /
                                                         sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::GT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                          (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                          (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
                                                         sizeof(bfloat16_t));
}
TEST_F(TestApiCompareUT, CompareCount_Gt_int8_uint8) {
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GT>(ONE_BLK_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GT>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) /
                                                     sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::GT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                      (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                      (ONE_BLK_SIZE - sizeof(int8_t))) /
                                                     sizeof(int8_t));
}
TEST_F(TestApiCompareUT, CompareCount_Gt_int16_uint8) {
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GT>(ONE_BLK_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GT>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) /
                                                      sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::GT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                       (ONE_BLK_SIZE - sizeof(int16_t))) /
                                                      sizeof(int16_t));
}
TEST_F(TestApiCompareUT, CompareCount_Gt_uint8_uint8) {
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>(ONE_BLK_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) /
                                                      sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::GT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                       (ONE_BLK_SIZE - sizeof(uint8_t))) /
                                                      sizeof(uint8_t));
}
// Lt
TEST_F(TestApiCompareUT, CompareCount_Lt_bfloat_uint8) {
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>(ONE_BLK_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) /
                                                         sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE /
                                                         sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                          (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                          (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
                                                         sizeof(bfloat16_t));
}
TEST_F(TestApiCompareUT, CompareCount_Lt_int8_uint8) {
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LT>(ONE_BLK_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LT>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) /
                                                     sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                      (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                      (ONE_BLK_SIZE - sizeof(int8_t))) /
                                                     sizeof(int8_t));
}
TEST_F(TestApiCompareUT, CompareCount_Lt_int16_uint8) {
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LT>(ONE_BLK_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LT>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) /
                                                      sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                       (ONE_BLK_SIZE - sizeof(int16_t))) /
                                                      sizeof(int16_t));
}
TEST_F(TestApiCompareUT, CompareCount_Lt_uint8_uint8) {
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LT>(ONE_BLK_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LT>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LT>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LT>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) /
                                                      sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LT>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LT>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                       (ONE_BLK_SIZE - sizeof(uint8_t))) /
                                                      sizeof(uint8_t));
}
// Le
TEST_F(TestApiCompareUT, CompareCount_Le_bfloat_uint8) {
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>(ONE_BLK_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) /
                                                         sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE /
                                                         sizeof(bfloat16_t));
  TensorCompareTest<uint8_t, bfloat16_t, 1, CMPMODE::LE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                          (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                          (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
                                                         sizeof(bfloat16_t));
}
TEST_F(TestApiCompareUT, CompareCount_Le_int8_uint8) {
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LE>(ONE_BLK_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LE>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) /
                                                     sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTest<uint8_t, int8_t, 1, CMPMODE::LE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                      (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                      (ONE_BLK_SIZE - sizeof(int8_t))) /
                                                     sizeof(int8_t));
}
TEST_F(TestApiCompareUT, CompareCount_Le_int16_uint8) {
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LE>(ONE_BLK_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LE>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) /
                                                      sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTest<uint8_t, int16_t, 1, CMPMODE::LE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                       (ONE_BLK_SIZE - sizeof(int16_t))) /
                                                      sizeof(int16_t));
}
TEST_F(TestApiCompareUT, CompareCount_Le_uint8_uint8) {
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>(ONE_BLK_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) /
                                                      sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTest<uint8_t, uint8_t, 1, CMPMODE::LE>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE +
                                                       (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                                                       (ONE_BLK_SIZE - sizeof(uint8_t))) /
                                                      sizeof(uint8_t));
}

// compare的normal模式
TEST_F(TestApiCompareUT, CompareNormal_Eq_float_uint8) {
  TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(2, ONE_BLK_SIZE / sizeof(float));
  TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(3, ONE_REPEAT_BYTE_SIZE / sizeof(float));
  TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(float) / 4);
  TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(float)) / sizeof(float));
  TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(float) / 6);
  TensorCompareTestNormal<uint8_t, float, 2, CMPMODE::EQ>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(float))) /
             sizeof(float) / 8);
}
// GE
TEST_F(TestApiCompareUT, CompareNormal_Ge_bfloat_uint8) {
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GE>(2, ONE_BLK_SIZE / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GE>(3, ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GE>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t) / 4);
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GE>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GE>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t) / 6);
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GE>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
             sizeof(bfloat16_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Ge_int8_uint8) {
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GE>(2, ONE_BLK_SIZE / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GE>(3, ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GE>(4,
                                                           MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t) / 4);
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GE>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GE>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t) / 6);
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GE>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(int8_t))) /
             sizeof(int8_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Ge_int16_uint8) {
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GE>(2, ONE_BLK_SIZE / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GE>(3, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GE>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t) / 4);
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GE>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GE>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t) / 6);
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GE>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(int16_t))) /
             sizeof(int16_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Ge_uint8_uint8) {
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GE>(2, ONE_BLK_SIZE / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GE>(3, ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GE>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t) / 4);
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GE>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GE>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t) / 6);
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GE>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(uint8_t))) /
             sizeof(uint8_t) / 8);
}
// Gt
TEST_F(TestApiCompareUT, CompareNormal_Gt_bfloat_uint8) {
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GT>(2, ONE_BLK_SIZE / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GT>(3, ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GT>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t) / 4);
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GT>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GT>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t) / 6);
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::GT>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
             sizeof(bfloat16_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Gt_int8_uint8) {
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GT>(2, ONE_BLK_SIZE / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GT>(3, ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GT>(4,
                                                           MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t) / 4);
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GT>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GT>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t) / 6);
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::GT>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(int8_t))) /
             sizeof(int8_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Gt_int16_uint8) {
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GT>(2, ONE_BLK_SIZE / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GT>(3, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GT>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t) / 4);
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GT>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GT>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t) / 6);
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::GT>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(int16_t))) /
             sizeof(int16_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Gt_uint8_uint8) {
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GT>(2, ONE_BLK_SIZE / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GT>(3, ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GT>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t) / 4);
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GT>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GT>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t) / 6);
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::GT>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(uint8_t))) /
             sizeof(uint8_t) / 8);
}
// Lt
TEST_F(TestApiCompareUT, CompareNormal_Lt_bfloat_uint8) {
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LT>(2, ONE_BLK_SIZE / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LT>(3, ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LT>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t) / 4);
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LT>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LT>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t) / 6);
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LT>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
             sizeof(bfloat16_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Lt_int8_uint8) {
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LT>(2, ONE_BLK_SIZE / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LT>(3, ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LT>(4,
                                                           MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t) / 4);
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LT>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LT>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t) / 6);
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LT>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(int8_t))) /
             sizeof(int8_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Lt_int16_uint8) {
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LT>(2, ONE_BLK_SIZE / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LT>(3, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LT>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t) / 4);
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LT>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LT>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t) / 6);
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LT>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(int16_t))) /
             sizeof(int16_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Lt_uint8_uint8) {
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LT>(2, ONE_BLK_SIZE / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LT>(3, ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LT>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t) / 4);
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LT>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LT>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t) / 6);
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LT>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(uint8_t))) /
             sizeof(uint8_t) / 8);
}
// Le
TEST_F(TestApiCompareUT, CompareNormal_Le_bfloat_uint8) {
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LE>(2, ONE_BLK_SIZE / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LE>(3, ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LE>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t) / 4);
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LE>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LE>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t) / 6);
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::LE>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
             sizeof(bfloat16_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Le_int8_uint8) {
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LE>(2, ONE_BLK_SIZE / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LE>(3, ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LE>(4,
                                                           MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t) / 4);
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LE>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LE>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t) / 6);
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::LE>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(int8_t))) /
             sizeof(int8_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Le_int16_uint8) {
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LE>(2, ONE_BLK_SIZE / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LE>(3, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LE>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t) / 4);
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LE>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LE>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t) / 6);
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::LE>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(int16_t))) /
             sizeof(int16_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Le_uint8_uint8) {
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LE>(2, ONE_BLK_SIZE / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LE>(3, ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LE>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t) / 4);
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LE>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LE>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t) / 6);
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::LE>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(uint8_t))) /
             sizeof(uint8_t) / 8);
}
// Eq
TEST_F(TestApiCompareUT, CompareNormal_Eq_bfloat_uint8) {
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::EQ>(2, ONE_BLK_SIZE / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::EQ>(3, ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::EQ>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t) / 4);
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::EQ>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(bfloat16_t)) / sizeof(bfloat16_t));
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::EQ>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t) / 6);
  TensorCompareTestNormal<uint8_t, bfloat16_t, 1, CMPMODE::EQ>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(bfloat16_t))) /
             sizeof(bfloat16_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Eq_int8_uint8) {
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::EQ>(2, ONE_BLK_SIZE / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::EQ>(3, ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::EQ>(4,
                                                           MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t) / 4);
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::EQ>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int8_t)) / sizeof(int8_t));
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::EQ>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t) / 6);
  TensorCompareTestNormal<uint8_t, int8_t, 1, CMPMODE::EQ>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(int8_t))) /
             sizeof(int8_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Eq_uint8_uint8) {
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::EQ>(2, ONE_BLK_SIZE / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::EQ>(3, ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::EQ>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t) / 4);
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::EQ>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(uint8_t)) / sizeof(uint8_t));
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::EQ>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t) / 6);
  TensorCompareTestNormal<uint8_t, uint8_t, 1, CMPMODE::EQ>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(uint8_t))) /
             sizeof(uint8_t) / 8);
}
TEST_F(TestApiCompareUT, CompareNormal_Eq_int16_uint8) {
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::EQ>(2, ONE_BLK_SIZE / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::EQ>(3, ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::EQ>(
      4, MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t) / 4);
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::EQ>(
      5, (ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE + sizeof(int16_t)) / sizeof(int16_t));
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::EQ>(
      6, (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int16_t) / 6);
  TensorCompareTestNormal<uint8_t, int16_t, 1, CMPMODE::EQ>(
      7, ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
          (ONE_BLK_SIZE - sizeof(int16_t))) /
             sizeof(int16_t) / 8);
}

// bool 输入（底层按 uint8 执行），输出 uint8（比较结果 0/1，位模式一致）。
// BOOL 与 uint8 共享全部内部路径，shape 边界由 uint8 用例覆盖，此处每种模式仅取 1 组最小 shape
TEST_F(TestApiCompareUT, CompareCount_Le_bool_bool) {
  TensorCompareTest<uint8_t, bool, 1, CMPMODE::LE>(ONE_BLK_SIZE / sizeof(bool));
}

TEST_F(TestApiCompareUT, CompareScalar_Le_bool_bool) {
  // src1 取 0：期望 = (src0 截断值 <= 0)，src0=0 时 true、src0=1 时 false，两种输出均有覆盖
  CompareTest<uint8_t, bool, 1, CMPMODE::LE>(ONE_BLK_SIZE / sizeof(bool), 0);
}

TEST_F(TestApiCompareUT, CompareNormal_Le_bool_bool) {
  // def_src1 取 0：dim=2 且 size 为一个 block 时 src1 会被广播为 blk tensor，
  // 若 src1[0]=1 则期望恒 true，取 0 保证期望有 true/false 区分度
  TensorCompareTestNormal<uint8_t, bool, 2, CMPMODE::LE>(2, ONE_BLK_SIZE / sizeof(bool), 0);
}
}  // namespace af
