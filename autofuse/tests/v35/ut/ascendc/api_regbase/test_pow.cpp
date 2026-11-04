/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/pow.h"

using namespace AscendC;

template <typename T>
struct PowInputParam {
  T *y{};
  T *x1{};
  T *x2{};
  T *exp{};
  uint32_t size{};
};

class TestApiPowUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename T>
  static void InvokeTensorTensorKernel(PowInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_x2 = x2buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.x1, param.size);
    GmToUb(l_x2, param.x2, param.size);
    Pow(l_y, l_x1, l_x2, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(PowInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x2 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
      std::uniform_real_distribution<float> data_distr(0.0f, 10.0f);
      std::uniform_real_distribution<float> pow_distr(0.0f, 2.0f);
      param.x1[i] = static_cast<T>(data_distr(eng));
      param.x2[i] = static_cast<T>(pow_distr(eng));
      param.exp[i] = static_cast<T>(std::pow(static_cast<float>(param.x1[i]), static_cast<float>(param.x2[i])));
    }
  }

  template <typename T>
  static uint32_t Valid(PowInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      const double atol = 1e-3;
      const double delta = static_cast<double>(param.y[i]) - static_cast<double>(param.exp[i]);
      if (delta > atol && delta < (-atol)) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void FreeTensorInput(PowInputParam<T> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.x1);
    AscendC::GmFree(param.x2);
    AscendC::GmFree(param.exp);
  }

  // Tensor - Tensor 测试
  template <typename T>
  static void PowTensorTensorTest(uint32_t size) {
    PowInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);

    // 释放内存
    FreeTensorInput(param);
  }

  // Tensor - Scalar 场景：tensor底数 + float整数指数（5.0/-2.0/7.0 不在0/1/-1/2/0.5/-0.5/3.0特判内）
  template <typename T>
  static void InvokeTensorScalarKernel(PowInputParam<T> &param, const T &exp_value, bool use_fast) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, ybuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.x1, param.size);
    if (use_fast) {
      PowFast(l_y, l_x1, exp_value, l_tmp, param.size);
    } else {
      Pow(l_y, l_x1, exp_value, l_tmp, param.size);
    }
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorScalarInput(PowInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x2 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);
    for (uint32_t i = 0; i < param.size; i++) {
      std::uniform_real_distribution<float> data_distr(0.5f, 4.0f);
      param.x1[i] = static_cast<T>(data_distr(eng));
    }
  }

  // Tensor - Scalar 精度校验：相对误差为主，绝对误差兜底
  template <typename T>
  static uint32_t ValidTensorScalar(PowInputParam<T> &param, const T &exp_value) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      const double expected = std::pow(static_cast<double>(param.x1[i]), static_cast<double>(exp_value));
      const double actual = static_cast<double>(param.y[i]);
      const double atol = 1e-3;
      const double rtol = 1e-4;
      if (std::fabs(actual - expected) > (atol + rtol * std::fabs(expected))) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void FreeTensorScalarInput(PowInputParam<T> &param) {
    FreeTensorInput(param);
  }

  // Tensor - Scalar 测试：exp为float非特判整数，分别验证Pow高精度路径与PowFast快速幂路径
  template <typename T>
  static void PowTensorScalarTest(uint32_t size, const T &exp_value, bool use_fast) {
    PowInputParam<T> param{};
    param.size = size;
    CreateTensorScalarInput(param);

    auto kernel = [&param, &exp_value, use_fast] { InvokeTensorScalarKernel(param, exp_value, use_fast); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    EXPECT_EQ(ValidTensorScalar(param, exp_value), 0);

    FreeTensorScalarInput(param);
  }

  // Scalar - Scalar 场景：验证scratch从tmp_buf前两块之后偏移传入时负底数结果的正确性。
  // tmp按仓内UT惯例分配TMP_UB_SIZE（不小于CalcPowTmpSizeV2全标量分配量：dtypesize*1+32*2+scratch
  // 384B），分配公式本身由reg_func_v2的CalcPowTmpSizeV2Test断言覆盖。
  template <typename T>
  static void InvokeScalarScalarKernel(PowInputParam<T> &param, const T &base, const T &exp_value, bool use_fast) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> ybuf, tmp;
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    if (use_fast) {
      PowFast(l_y, base, exp_value, l_tmp, param.size);
    } else {
      Pow(l_y, base, exp_value, l_tmp, param.size);
    }
    for (uint32_t i = 0; i < param.size; i++) {
      param.y[i] = l_y.GetValue(i);
    }
  }

  // Scalar - Scalar 精度校验：负底数+非整数指数期望NaN，其余按std::pow基准
  template <typename T>
  static void ValidScalarScalar(PowInputParam<T> &param, const T &base, const T &exp_value) {
    const double expected = std::pow(static_cast<double>(base), static_cast<double>(exp_value));
    for (uint32_t i = 0; i < param.size; i++) {
      const double actual = static_cast<double>(param.y[i]);
      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual));
      } else {
        const double atol = 1e-3;
        const double rtol = 1e-4;
        EXPECT_NEAR(actual, expected, atol + rtol * std::fabs(expected));
      }
    }
  }

  // Scalar - Scalar 测试：指数取5.0/2.5（非特判），覆盖正/负底数与Pow/PowFast两条路径
  template <typename T>
  static void PowScalarScalarTest(const T &base, const T &exp_value, bool use_fast) {
    PowInputParam<T> param{};
    param.size = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    auto kernel = [&param, &base, &exp_value, use_fast] { InvokeScalarScalarKernel(param, base, exp_value, use_fast); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    ValidScalarScalar(param, base, exp_value);

    AscendC::GmFree(param.y);
  }
};

// ============ Tensor - Tensor 测试 ============
TEST_F(TestApiPowUT, Pow_TensorTensor_Test) {
  // int8
  PowTensorTensorTest<int8_t>(ONE_BLK_SIZE / sizeof(int8_t));
  PowTensorTensorTest<int8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  PowTensorTensorTest<int8_t>((ONE_BLK_SIZE - sizeof(int8_t)) / sizeof(int8_t));
  PowTensorTensorTest<int8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int8_t));
  PowTensorTensorTest<int8_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));
  PowTensorTensorTest<int8_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));

  // int16
  PowTensorTensorTest<int16_t>(ONE_BLK_SIZE / sizeof(int16_t));
  PowTensorTensorTest<int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  PowTensorTensorTest<int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
  PowTensorTensorTest<int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t));
  PowTensorTensorTest<int16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));
  PowTensorTensorTest<int16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));

  // int32
  PowTensorTensorTest<int>(ONE_BLK_SIZE / sizeof(int));
  PowTensorTensorTest<int>(ONE_REPEAT_BYTE_SIZE / sizeof(int));
  PowTensorTensorTest<int>((ONE_BLK_SIZE - sizeof(int)) / sizeof(int));
  PowTensorTensorTest<int>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int));
  PowTensorTensorTest<int>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int));
  PowTensorTensorTest<int>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int));

  // uint8
  PowTensorTensorTest<uint8_t>(ONE_BLK_SIZE / sizeof(uint8_t));
  PowTensorTensorTest<uint8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  PowTensorTensorTest<uint8_t>((ONE_BLK_SIZE - sizeof(uint8_t)) / sizeof(uint8_t));
  PowTensorTensorTest<uint8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint8_t));
  PowTensorTensorTest<uint8_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint8_t));
  PowTensorTensorTest<uint8_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint8_t));

  // uint16
  PowTensorTensorTest<uint16_t>(ONE_BLK_SIZE / sizeof(uint16_t));
  PowTensorTensorTest<uint16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t));
  PowTensorTensorTest<uint16_t>((ONE_BLK_SIZE - sizeof(uint16_t)) / sizeof(uint16_t));
  PowTensorTensorTest<uint16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint16_t));
  PowTensorTensorTest<uint16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint16_t));
  PowTensorTensorTest<uint16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint16_t));

  // uint32
  PowTensorTensorTest<uint32_t>(ONE_BLK_SIZE / sizeof(uint32_t));
  PowTensorTensorTest<uint32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
  PowTensorTensorTest<uint32_t>((ONE_BLK_SIZE - sizeof(uint32_t)) / sizeof(uint32_t));
  PowTensorTensorTest<uint32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint32_t));
  PowTensorTensorTest<uint32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t));
  PowTensorTensorTest<uint32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t));

  // float
  PowTensorTensorTest<float>(ONE_BLK_SIZE / sizeof(float));
  PowTensorTensorTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  PowTensorTensorTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  PowTensorTensorTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  PowTensorTensorTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  PowTensorTensorTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));

  // half
  PowTensorTensorTest<half>(ONE_BLK_SIZE / sizeof(half));
  PowTensorTensorTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  PowTensorTensorTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  PowTensorTensorTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  PowTensorTensorTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  PowTensorTensorTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));

  // BF16
  PowTensorTensorTest<bfloat16_t>(ONE_BLK_SIZE / sizeof(bfloat16_t));
  PowTensorTensorTest<bfloat16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  PowTensorTensorTest<bfloat16_t>((ONE_BLK_SIZE - sizeof(bfloat16_t)) / sizeof(bfloat16_t));
  PowTensorTensorTest<bfloat16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(bfloat16_t));
  PowTensorTensorTest<bfloat16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(bfloat16_t));
  PowTensorTensorTest<bfloat16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(bfloat16_t));
}

// ============ Tensor - Scalar 测试（float 非特判整数指数） ============
// 指数取 5.0f/-2.0f，避开 0/1/-1/2/0.5/-0.5/3.0 特判分支，分别验证 Power 高精度路径与快速幂路径
TEST_F(TestApiPowUT, Pow_TensorScalar_Test) {
  PowTensorScalarTest<float>(ONE_BLK_SIZE / sizeof(float), 5.0f, false);
  PowTensorScalarTest<float>(ONE_BLK_SIZE / sizeof(float), 5.0f, true);
  PowTensorScalarTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), -2.0f, false);
  PowTensorScalarTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), -2.0f, true);
}

// ============ Scalar - Scalar 测试（负底数修正路径） ============
// scalar+scalar场景下src1_buf(底数)复用tmp_buf前两块，scratch需从前两块之后偏移传入
// （left_tmp_buf），否则与src1_buf同址覆盖底数，负底数经Power的SpecialCase修正失效；
// 指数5.0/2.5避开特判分支。Pow走125行DOUBLE_FLOAT_TECH；PowFast整数指数走快速幂(对照)，
// 非整数走282行同形态Power。
TEST_F(TestApiPowUT, Pow_ScalarScalar_Test) {
  // 正底数：主计算链仅依赖scratch，预期正确
  PowScalarScalarTest<float>(2.0f, 5.0f, false);
  PowScalarScalarTest<float>(2.0f, 5.0f, true);
  // 负底数+奇整数指数：期望-32/-128，若scratch覆盖src1_buf则丢负号
  PowScalarScalarTest<float>(-2.0f, 5.0f, false);
  PowScalarScalarTest<float>(-2.0f, 7.0f, true);
  // 负底数+非整数指数：期望NaN，若scratch覆盖src1_buf则输出正数
  PowScalarScalarTest<float>(-2.0f, 2.5f, false);
  PowScalarScalarTest<float>(-2.0f, 2.5f, true);
}
