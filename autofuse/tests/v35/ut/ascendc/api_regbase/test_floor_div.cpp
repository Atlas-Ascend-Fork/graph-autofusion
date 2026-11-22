/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <random>
#include <type_traits>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/floor_div.h"

using namespace AscendC;

template <typename T>
struct TensorFloorDivInputParam {
  T *y{};
  T *exp{};
  T *src0{};
  T *src1{};
  uint32_t size{0};
  uint32_t out_size{0};
};

class TestRegbaseApiFloorDiv : public testing::Test {
 protected:
  template <typename T>
  static void InvokeKernelWithTwoTensorInput(TensorFloorDivInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;

    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, 65312);

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_x2 = x2buf.Get<T>();

    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, param.size);
    GmToUb(l_x2, param.src1, param.size);
    FloorDivExtend(l_y, l_x1, l_x2, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(TensorFloorDivInputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    int input_range = 10;

    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      // 有符号整型：src0 范围 [-input_range, input_range]，src1 范围 [-input_range, input_range]（排除 0）
      std::mt19937 eng(1);
      std::uniform_int_distribution<int32_t> distr(-input_range, input_range);
      std::mt19937 eng1(3);
      std::uniform_int_distribution<int32_t> distr1(-input_range, input_range);

      for (int i = 0; i < param.size; i++) {
        int32_t input = distr(eng);
        int32_t input1 = distr1(eng1);
        while (input1 == 0) {
          input1 = distr1(eng1);
        }
        param.src0[i] = static_cast<T>(input);
        param.src1[i] = static_cast<T>(input1);
        param.exp[i] = static_cast<T>(std::floor(static_cast<double>(input) / static_cast<double>(input1)));
      }
    } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
      // 无符号整型：src0 范围 [0, input_range]，src1 范围 [1, input_range]
      std::mt19937 eng(1);
      std::uniform_int_distribution<uint32_t> distr(0, input_range);
      std::mt19937 eng1(3);
      std::uniform_int_distribution<uint32_t> distr1(1, input_range);

      for (int i = 0; i < param.size; i++) {
        uint32_t input = distr(eng);
        uint32_t input1 = distr1(eng1);
        param.src0[i] = static_cast<T>(input);
        param.src1[i] = static_cast<T>(input1);
        param.exp[i] = static_cast<T>(std::floor(static_cast<double>(input) / static_cast<double>(input1)));
      }
    } else {
      // 浮点类型（half/float）：原始逻辑
      std::mt19937 eng(1);
      std::uniform_int_distribution distr(0, input_range);  // Define the range

      // 构造src1的随机生成器
      std::mt19937 eng1(3);                                  // Seed the generator
      std::uniform_int_distribution distr1(1, input_range);  // Define the range

      for (int i = 0; i < param.size; i++) {
        T input = distr(eng);  // Use the secure random number generator
        T input1 = distr1(eng1);
        param.src0[i] = input;
        param.src1[i] = input1;
        param.exp[i] = std::floor((double)input / (double)input1);
      }
    }
  }

  template <typename T>
  static uint32_t Valid(T *y, T *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (y[i] != exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void FreeTensorInput(TensorFloorDivInputParam<T> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
    AscendC::GmFree(param.src1);
  }

  template <typename T>
  static void FloorDivTest(uint32_t size) {
    TensorFloorDivInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernelWithTwoTensorInput(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);

    // 释放内存
    FreeTensorInput(param);
  }
};

TEST_F(TestRegbaseApiFloorDiv, FloorDiv_Test) {
  FloorDivTest<half>(ONE_BLK_SIZE / sizeof(half));
  FloorDivTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  FloorDivTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  FloorDivTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  FloorDivTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  FloorDivTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  FloorDivTest<float>(ONE_BLK_SIZE / sizeof(float));
  FloorDivTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  FloorDivTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  FloorDivTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  FloorDivTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  FloorDivTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                       (ONE_BLK_SIZE - sizeof(float))) /
                      2 / sizeof(float));
}

TEST_F(TestRegbaseApiFloorDiv, FloorDiv_Int16_Test) {
  FloorDivTest<int16_t>(ONE_BLK_SIZE / sizeof(int16_t));
  FloorDivTest<int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  FloorDivTest<int16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));
  FloorDivTest<int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
  FloorDivTest<int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t));
  FloorDivTest<int16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));
}

TEST_F(TestRegbaseApiFloorDiv, FloorDiv_UInt16_Test) {
  FloorDivTest<uint16_t>(ONE_BLK_SIZE / sizeof(uint16_t));
  FloorDivTest<uint16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t));
  FloorDivTest<uint16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint16_t));
  FloorDivTest<uint16_t>((ONE_BLK_SIZE - sizeof(uint16_t)) / sizeof(uint16_t));
  FloorDivTest<uint16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint16_t));
  FloorDivTest<uint16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint16_t));
}

TEST_F(TestRegbaseApiFloorDiv, FloorDiv_Int32_Test) {
  FloorDivTest<int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  FloorDivTest<int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  FloorDivTest<int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));
  FloorDivTest<int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  FloorDivTest<int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  FloorDivTest<int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));
}

TEST_F(TestRegbaseApiFloorDiv, FloorDiv_Int64_Test) {
  FloorDivTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t));
  FloorDivTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  FloorDivTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t));
}

TEST_F(TestRegbaseApiFloorDiv, FloorDiv_UInt32_Test) {
  FloorDivTest<uint32_t>(ONE_BLK_SIZE / sizeof(uint32_t));
  FloorDivTest<uint32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
  FloorDivTest<uint32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t));
  FloorDivTest<uint32_t>((ONE_BLK_SIZE - sizeof(uint32_t)) / sizeof(uint32_t));
  FloorDivTest<uint32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint32_t));
  FloorDivTest<uint32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t));
}
