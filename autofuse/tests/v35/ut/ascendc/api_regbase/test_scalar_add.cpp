/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ui16 / ui32 的 tensor-scalar 包装 `AddsExtend`（`api_regbase/scalar_add.h`）：连续 LocalTensor
// 接口 `AscendC::Adds` 不收这两种类型，包装里是寄存器标量接口 `MicroAPI::Adds` 的循环。参考值在
// 宿主侧独立算，不借用被测接口。
//
// 缓冲区不按逻辑长度分：包装的载入是不带掩码的整寄存器载入，尾块那一轮会读满一个向量长度，
// 所以输入按整向量补齐，读不越界；输出在补齐长度之后多留一个向量长度做守卫区，预填哨兵，
// 调用后断言守卫区一个都没被改写。掩码同时作用于运算与存回，尾块外的 lane 靠它挡住——
// 只比对 [0, size) 的话，掩码退化成全 1 时存回写穿尾部而数值仍然全对，测不到这类回归。

#include <cstdint>
#include <limits>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/scalar_add.h"

using namespace AscendC;

namespace {

template <typename T>
struct ScalarBinaryParam {
  T *y{};
  T *exp{};
  T *src{};
  uint32_t size{0};
  uint32_t paddedSize{0};
  uint32_t guardedSize{0};
  T scalar{0};
};

template <typename T>
constexpr T kGuardSentinel = static_cast<T>(sizeof(T) == 2 ? 0xA5A5U : 0xA5A5A5A5U);

class TestRegbaseApiScalarAdd : public testing::Test {
 protected:
  template <typename T>
  static constexpr uint32_t VecElm() {
    return ONE_REPEAT_BYTE_SIZE / sizeof(T);
  }

  template <typename T>
  static void InvokeKernel(ScalarBinaryParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.paddedSize);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.guardedSize);

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();

    GmToUb(l_x, param.src, param.paddedSize);
    // 输出整块（含守卫区）先进 UB，守卫区因此在调用前是哨兵；调用后整块搬回 GM 供比对。
    GmToUb(l_y, param.y, param.guardedSize);
    AddsExtend(l_y, l_x, param.scalar, param.size);
    UbToGm(param.y, l_y, param.guardedSize);
  }

  // 一半输入贴着上限（加上标量后回绕到小值），一半是普通小值。
  template <typename T>
  static void CreateTensorInput(ScalarBinaryParam<T> &param) {
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.paddedSize));
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.guardedSize));

    // 输入的补齐段填 0，输出整块（含守卫区）预填哨兵。
    for (uint32_t i = 0; i < param.paddedSize; i++) {
      param.src[i] = static_cast<T>(0);
    }
    for (uint32_t i = 0; i < param.guardedSize; i++) {
      param.y[i] = kGuardSentinel<T>;
    }

    const uint32_t scalar = param.scalar;
    const uint32_t upper = std::numeric_limits<T>::max();
    for (uint32_t i = 0; i < param.size; i++) {
      const uint32_t value = (i % 2U == 0U) ? upper - (i % 8U) : (i * 7U) % 1024U;
      param.src[i] = static_cast<T>(value);
      param.exp[i] = static_cast<T>(value + scalar);
    }
  }

  template <typename T>
  static void FreeTensorInput(ScalarBinaryParam<T> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src);
  }

  template <typename T>
  static void InitSizes(ScalarBinaryParam<T> &param, uint32_t size) {
    param.size = size;
    // 不足一个向量的也要占满一个向量，载入才读不越界。
    param.paddedSize = AlignUp(size, VecElm<T>());
    param.guardedSize = param.paddedSize + VecElm<T>();
  }

  // 尺寸铺到不足一个块、刚好一个块、一个寄存器、带尾块的多轮、达到最大 repeat 数的整块。
  template <typename T>
  static void AddsExtendTest(T scalar) {
    const uint32_t sizes[] = {
        1U,
        (ONE_BLK_SIZE - sizeof(T)) / sizeof(T),
        ONE_BLK_SIZE / sizeof(T),
        ONE_REPEAT_BYTE_SIZE / sizeof(T),
        (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(T),
        ONE_REPEAT_BYTE_SIZE / sizeof(T) * 3U + 5U,
        MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(T),
    };
    for (const uint32_t size : sizes) {
      ScalarBinaryParam<T> param{};
      InitSizes(param, size);
      param.scalar = scalar;
      CreateTensorInput(param);

      // 构造Api调用函数
      auto kernel = [&param] { InvokeKernel(param); };

      // 调用kernel
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(kernel, 1);

      // 验证结果：[0, size) 与参考值逐个相等；[size, guardedSize) 仍是哨兵。
      uint32_t diff_count = Valid(param.y, param.exp, param.size);
      EXPECT_EQ(diff_count, 0U) << "size=" << size;
      uint32_t guard_diff_count = 0;
      for (uint32_t i = param.size; i < param.guardedSize; i++) {
        if (param.y[i] != kGuardSentinel<T>) {
          guard_diff_count++;
        }
      }
      EXPECT_EQ(guard_diff_count, 0U) << "size=" << size;

      // 释放内存
      FreeTensorInput(param);
    }
  }
};

TEST_F(TestRegbaseApiScalarAdd, AddsExtend_UInt16_Test) {
  AddsExtendTest<uint16_t>(static_cast<uint16_t>(3U));
  AddsExtendTest<uint16_t>(static_cast<uint16_t>(65535U));
}

TEST_F(TestRegbaseApiScalarAdd, AddsExtend_UInt32_Test) {
  AddsExtendTest<uint32_t>(static_cast<uint32_t>(3U));
  AddsExtendTest<uint32_t>(static_cast<uint32_t>(4294967295U));
}

}  // namespace
