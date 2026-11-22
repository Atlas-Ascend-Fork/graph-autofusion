/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <type_traits>
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/shift.h"

using namespace AscendC;

template <typename T>
struct TensorShiftInputParam {
  T *y{};
  T *exp{};
  T *src0{};  // x
  T *src1{};  // shift amount
  uint32_t size{0};
};

// 异型组合:x 为 T1,移位量为 ST(注册表允许 unsigned x + signed shift)
template <typename T1, typename ST>
struct TensorShiftMixedParam {
  T1 *y{};
  T1 *exp{};
  T1 *src0{};
  ST *src1{};
  uint32_t size{0};
};

// 按 eager(x86/pytorch)语义计算期望:left 负/越界→0;right signed/unsigned 移位量出界(s<0 或 s>=bits)统一按移
// bits-1(符号提取)
template <typename T>
static T RefLeftShift(T x, T shift) {
  constexpr int64_t bits = static_cast<int64_t>(sizeof(T) * 8);
  const int64_t s = static_cast<int64_t>(shift);
  if (s < 0 || s >= bits) {
    return static_cast<T>(0);
  }
  const uint64_t ux = static_cast<uint64_t>(static_cast<typename std::make_unsigned<T>::type>(x));
  const uint64_t r = ux << static_cast<uint64_t>(s);
  return static_cast<T>(static_cast<typename std::make_unsigned<T>::type>(r));
}

template <typename T>
static T RefRightShift(T x, T shift) {
  constexpr int64_t bits = static_cast<int64_t>(sizeof(T) * 8);
  const int64_t s = static_cast<int64_t>(shift);
  if constexpr (std::is_signed<T>::value) {
    // 对齐 x86/pytorch eager:移位量出界(s<0 或 s>=bits)统一按移 bits-1(符号提取:正数->0,负数->-1)
    const int64_t clamped = (s < 0 || s > bits - 1) ? bits - 1 : s;
    return static_cast<T>(static_cast<int64_t>(x) >> clamped);
  } else {
    const int64_t clamped = (s < 0 || s >= bits) ? bits - 1 : s;
    return static_cast<T>(static_cast<uint64_t>(x) >> static_cast<uint64_t>(clamped));
  }
}

// 异型组合的 right 期望:移位量按 ST(signed)语义修正,出界(s<0 或 s>=bits)统一按移 bits-1
template <typename T1, typename ST>
static T1 RefRightShiftMixed(T1 x, ST shift) {
  constexpr int64_t bits = static_cast<int64_t>(sizeof(T1) * 8);
  const int64_t s = static_cast<int64_t>(shift);
  const int64_t clamped = (s < 0 || s > bits - 1) ? bits - 1 : s;
  return static_cast<T1>(static_cast<uint64_t>(x) >> static_cast<uint64_t>(clamped));
}

class TestApiShift : public testing::Test {
 protected:
  template <bool IsLeft, typename T>
  static void InvokeKernel(TensorShiftInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x0buf, x1buf, ybuf;
    tpipe.InitBuffer(x0buf, sizeof(T) * param.size);
    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    LocalTensor<T> l_x0 = x0buf.Get<T>();
    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();

    GmToUb(l_x0, param.src0, param.size);
    GmToUb(l_x1, param.src1, param.size);
    if constexpr (IsLeft) {
      ShiftLeftExtend<T>(l_y, l_x0, l_x1, param.size);
    } else {
      ShiftRightExtend<T>(l_y, l_x0, l_x1, param.size);
    }
    UbToGm(param.y, l_y, param.size);
  }

  template <bool IsLeft, typename T>
  static void ShiftTest(const uint32_t size) {
    TensorShiftInputParam<T> param{};
    param.size = size;
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);
    std::uniform_int_distribution<int64_t> xDistr(-1000, 1000);
    // 全范围移位量:负/0/正常/越界正值(bits+2),期望值按各自修正语义精确断言
    const int64_t shiftUpper = static_cast<int64_t>(sizeof(T) * 8 + 2);
    std::uniform_int_distribution<int64_t> sDistr(-3, shiftUpper);
    for (uint32_t i = 0; i < param.size; i++) {
      param.src0[i] = static_cast<T>(xDistr(eng));
      param.src1[i] = static_cast<T>(sDistr(eng));
      param.exp[i] =
          IsLeft ? RefLeftShift<T>(param.src0[i], param.src1[i]) : RefRightShift<T>(param.src0[i], param.src1[i]);
    }

    auto kernel = [&param] { InvokeKernel<IsLeft, T>(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        diff_count++;
      }
    }
    EXPECT_EQ(diff_count, 0);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
    AscendC::GmFree(param.src1);
  }

  // 特征值:x 交错正负、shift 覆盖 0/正常/负/越界(含 min/全1),right 越界正值同样构造(clamp 语义可精确断言)
  template <bool IsLeft, typename T>
  static void ShiftTestWithValues() {
    constexpr int64_t bits = static_cast<int64_t>(sizeof(T) * 8);
    const int64_t shiftUpper = bits + 1;
    std::vector<T> xs;
    std::vector<T> shifts;
    for (int64_t s = -2; s <= shiftUpper; s++) {
      xs.push_back(static_cast<T>(1));
      shifts.push_back(static_cast<T>(s));
      xs.push_back(static_cast<T>(-1));
      shifts.push_back(static_cast<T>(s));
      xs.push_back(static_cast<T>(100));
      shifts.push_back(static_cast<T>(s));
    }
    xs.push_back(static_cast<T>(0));
    shifts.push_back(static_cast<T>(-1));
    xs.push_back(static_cast<T>(-100));
    shifts.push_back(static_cast<T>(0));

    TensorShiftInputParam<T> param{};
    param.size = static_cast<uint32_t>(xs.size());
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    for (uint32_t i = 0; i < param.size; i++) {
      param.src0[i] = xs[i];
      param.src1[i] = shifts[i];
      param.exp[i] = IsLeft ? RefLeftShift<T>(xs[i], shifts[i]) : RefRightShift<T>(xs[i], shifts[i]);
    }

    auto kernel = [&param] { InvokeKernel<IsLeft, T>(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        printf("mismatch idx=%d x=%d shift=%d expect=%d got=%d\n", i, static_cast<int>(param.src0[i]),
               static_cast<int>(param.src1[i]), static_cast<int>(param.exp[i]), static_cast<int>(param.y[i]));
        diff_count++;
      }
    }
    EXPECT_EQ(diff_count, 0);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
    AscendC::GmFree(param.src1);
  }

  // 异型组合:x 为 T1(signed/unsigned),移位量为 ST(signed),对齐注册表 {DT_UINTN, DT_INTN} 对。
  // 单独写调用逻辑(不经 ICPU_RUN_KF 包装的 lambda,直接同步调用),覆盖模板双参数推导路径
  template <bool IsLeft, typename T1, typename ST>
  static void ShiftMixedTest(const uint32_t size) {
    TensorShiftMixedParam<T1, ST> param{};
    param.size = size;
    param.y = static_cast<T1 *>(AscendC::GmAlloc(sizeof(T1) * param.size));
    param.exp = static_cast<T1 *>(AscendC::GmAlloc(sizeof(T1) * param.size));
    param.src0 = static_cast<T1 *>(AscendC::GmAlloc(sizeof(T1) * param.size));
    param.src1 = static_cast<ST *>(AscendC::GmAlloc(sizeof(ST) * param.size));

    std::mt19937 eng(3);
    std::uniform_int_distribution<int64_t> xDistr(0, 1000);  // unsigned x 用非负值
    // 全范围移位量:left 负/越界→0,right 负→0、越界→bits-1,均可精确断言
    const int64_t shiftUpper = static_cast<int64_t>(sizeof(T1) * 8 + 2);
    std::uniform_int_distribution<int64_t> sDistr(-3, shiftUpper);
    for (uint32_t i = 0; i < param.size; i++) {
      param.src0[i] = static_cast<T1>(xDistr(eng));
      param.src1[i] = static_cast<ST>(sDistr(eng));
      param.exp[i] = IsLeft ? RefLeftShift<T1>(param.src0[i], static_cast<T1>(param.src1[i]))
                            : RefRightShiftMixed<T1, ST>(param.src0[i], param.src1[i]);
    }

    auto kernel = [&param] {
      TPipe tpipe;
      TBuf<TPosition::VECCALC> x0buf, x1buf, ybuf;
      tpipe.InitBuffer(x0buf, sizeof(T1) * param.size);
      tpipe.InitBuffer(x1buf, sizeof(ST) * param.size);
      tpipe.InitBuffer(ybuf, sizeof(T1) * param.size);
      LocalTensor<T1> l_x0 = x0buf.Get<T1>();
      LocalTensor<ST> l_x1 = x1buf.Get<ST>();
      LocalTensor<T1> l_y = ybuf.Get<T1>();
      GmToUb(l_x0, param.src0, param.size);
      GmToUb(l_x1, param.src1, param.size);
      if constexpr (IsLeft) {
        ShiftLeftExtend<T1, ST>(l_y, l_x0, l_x1, param.size);
      } else {
        ShiftRightExtend<T1, ST>(l_y, l_x0, l_x1, param.size);
      }
      UbToGm(param.y, l_y, param.size);
    };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        printf("mismatch idx=%d x=%d shift=%d expect=%d got=%d\n", i, static_cast<int>(param.src0[i]),
               static_cast<int>(param.src1[i]), static_cast<int>(param.exp[i]), static_cast<int>(param.y[i]));
        diff_count++;
      }
    }
    EXPECT_EQ(diff_count, 0);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src0);
    AscendC::GmFree(param.src1);
  }
};

// 随机值:正常 + 负移位 + 越界,覆盖 8/16/32/64 位全部 signed/unsigned
TEST_F(TestApiShift, ShiftLeft_Test_int8) {
  ShiftTest<true, int8_t>(130);
}
TEST_F(TestApiShift, ShiftLeft_Test_uint8) {
  ShiftTest<true, uint8_t>(130);
}
TEST_F(TestApiShift, ShiftLeft_Test_int16) {
  ShiftTest<true, int16_t>(130);
}
TEST_F(TestApiShift, ShiftLeft_Test_uint16) {
  ShiftTest<true, uint16_t>(130);
}
TEST_F(TestApiShift, ShiftLeft_Test_int32) {
  ShiftTest<true, int32_t>(130);
}
TEST_F(TestApiShift, ShiftLeft_Test_uint32) {
  ShiftTest<true, uint32_t>(130);
}
TEST_F(TestApiShift, ShiftLeft_Test_int64) {
  ShiftTest<true, int64_t>(130);
}
TEST_F(TestApiShift, ShiftLeft_Test_uint64) {
  ShiftTest<true, uint64_t>(130);
}

TEST_F(TestApiShift, ShiftRight_Test_int8) {
  ShiftTest<false, int8_t>(130);
}
TEST_F(TestApiShift, ShiftRight_Test_uint8) {
  ShiftTest<false, uint8_t>(130);
}
TEST_F(TestApiShift, ShiftRight_Test_int16) {
  ShiftTest<false, int16_t>(130);
}
TEST_F(TestApiShift, ShiftRight_Test_uint16) {
  ShiftTest<false, uint16_t>(130);
}
TEST_F(TestApiShift, ShiftRight_Test_int32) {
  ShiftTest<false, int32_t>(130);
}
TEST_F(TestApiShift, ShiftRight_Test_uint32) {
  ShiftTest<false, uint32_t>(130);
}
TEST_F(TestApiShift, ShiftRight_Test_int64) {
  ShiftTest<false, int64_t>(130);
}
TEST_F(TestApiShift, ShiftRight_Test_uint64) {
  ShiftTest<false, uint64_t>(130);
}

// 特征值:每个 shift in [-2, bits+1] 对三个 x 值,锁死单算子语义的每个关键点
TEST_F(TestApiShift, ShiftLeft_Values_int8) {
  ShiftTestWithValues<true, int8_t>();
}
TEST_F(TestApiShift, ShiftLeft_Values_uint8) {
  ShiftTestWithValues<true, uint8_t>();
}
TEST_F(TestApiShift, ShiftLeft_Values_int16) {
  ShiftTestWithValues<true, int16_t>();
}
TEST_F(TestApiShift, ShiftLeft_Values_uint32) {
  ShiftTestWithValues<true, uint32_t>();
}
TEST_F(TestApiShift, ShiftLeft_Values_int64) {
  ShiftTestWithValues<true, int64_t>();
}
TEST_F(TestApiShift, ShiftLeft_Values_uint64) {
  ShiftTestWithValues<true, uint64_t>();
}
TEST_F(TestApiShift, ShiftRight_Values_int8) {
  ShiftTestWithValues<false, int8_t>();
}
TEST_F(TestApiShift, ShiftRight_Values_uint8) {
  ShiftTestWithValues<false, uint8_t>();
}
TEST_F(TestApiShift, ShiftRight_Values_int32) {
  ShiftTestWithValues<false, int32_t>();
}
TEST_F(TestApiShift, ShiftRight_Values_uint64) {
  ShiftTestWithValues<false, uint64_t>();
}

// 尾部 count:非整 repeat
TEST_F(TestApiShift, ShiftLeft_Tail_int32) {
  ShiftTest<true, int32_t>(67);
}
TEST_F(TestApiShift, ShiftLeft_Tail_uint64) {
  ShiftTest<true, uint64_t>(33);
}
TEST_F(TestApiShift, ShiftRight_Tail_int8) {
  ShiftTest<false, int8_t>(67);
}
TEST_F(TestApiShift, ShiftRight_Tail_uint32) {
  ShiftTest<false, uint32_t>(33);
}

// 异型组合(注册表的 {DT_UINTN, DT_INTN} 对):unsigned x + signed 移位量,编译期即锁死双模板参数推导
TEST_F(TestApiShift, ShiftLeft_Mixed_uint8_int8) {
  ShiftMixedTest<true, uint8_t, int8_t>(130);
}
TEST_F(TestApiShift, ShiftLeft_Mixed_uint16_int16) {
  ShiftMixedTest<true, uint16_t, int16_t>(130);
}
TEST_F(TestApiShift, ShiftLeft_Mixed_uint32_int32) {
  ShiftMixedTest<true, uint32_t, int32_t>(130);
}
TEST_F(TestApiShift, ShiftLeft_Mixed_uint64_int64) {
  ShiftMixedTest<true, uint64_t, int64_t>(130);
}
TEST_F(TestApiShift, ShiftRight_Mixed_uint8_int8) {
  ShiftMixedTest<false, uint8_t, int8_t>(130);
}
TEST_F(TestApiShift, ShiftRight_Mixed_uint16_int16) {
  ShiftMixedTest<false, uint16_t, int16_t>(130);
}
TEST_F(TestApiShift, ShiftRight_Mixed_uint32_int32) {
  ShiftMixedTest<false, uint32_t, int32_t>(130);
}
TEST_F(TestApiShift, ShiftRight_Mixed_uint64_int64) {
  ShiftMixedTest<false, uint64_t, int64_t>(130);
}
