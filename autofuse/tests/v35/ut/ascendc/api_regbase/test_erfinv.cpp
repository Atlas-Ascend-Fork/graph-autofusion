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
#include <algorithm>
#include <vector>
#include "gtest/gtest.h"
#include "boost/math/special_functions/erf.hpp"
#include "tikicpulib.h"
#include "simt_api/math_functions.h"
#include "test_api_utils.h"
#include "api_regbase/erfinv.h"

using namespace AscendC;

namespace af {

template <typename T>
struct TensorInputParam {
  T *dst{};
  T *exp{};
  T *src{};
  int32_t size{0};
};

class TestApiErfinv : public testing::Test {
 protected:
  template <typename T>
  static void InvokeTensorKernel(TensorInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> srcBuf, dstBuf;
    tpipe.InitBuffer(srcBuf, sizeof(T) * param.size);
    tpipe.InitBuffer(dstBuf, sizeof(T) * param.size);

    LocalTensor<T> l_src = srcBuf.Get<T>();
    LocalTensor<T> l_dst = dstBuf.Get<T>();

    GmToUb(l_src, param.src, param.size);
    ErfinvExtend<T>(l_dst, l_src, param.size);
    UbToGm(param.dst, l_dst, param.size);
  }

  template <typename T>
  static void CreateTensorInput(TensorInputParam<T> &param) {
    // 构造测试输入和预期结果
    param.dst = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    auto distr = []() {
      if constexpr (std::is_same_v<T, float>) {
        return std::uniform_real_distribution<float>(-1.0f, 1.0f);
      }
    }();

    for (int i = 0; i < param.size; i++) {
      T input = static_cast<T>(distr(eng));
      param.src[i] = input;
      param.exp[i] = boost::math::erf_inv(param.src[i]);
    }
  }

  template <typename T>
  static void CreateSpecialTensorInput(TensorInputParam<T> &param) {
    param.dst = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    T nan_val = std::numeric_limits<T>::quiet_NaN();
    T inf_val = std::numeric_limits<T>::infinity();

    param.src[0] = 0.0f;
    param.exp[0] = 0.0f;

    param.src[1] = -0.999f;
    param.exp[1] = -2.326758f;

    param.src[2] = 0.999f;
    param.exp[2] = 2.326758f;

    param.src[3] = 1.0f;
    param.exp[3] = inf_val;

    param.src[4] = -1.0f;
    param.exp[4] = -inf_val;

    param.src[5] = 1.01f;
    param.exp[5] = nan_val;

    param.src[6] = -1.01f;
    param.exp[6] = nan_val;

    param.src[7] = inf_val;
    param.exp[7] = nan_val;

    param.src[8] = -inf_val;
    param.exp[8] = nan_val;

    param.src[9] = nan_val;
    param.exp[9] = nan_val;
  }

  template <typename T>
  static uint32_t Valid(T *dst, T *exp, size_t comp_size, double tol = 1e-4) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (std::isnan(exp[i])) {
        if (!std::isnan(dst[i])) {
          diff_count++;
        }
      } else if (std::isinf(exp[i])) {
        if (dst[i] != exp[i]) {
          diff_count++;
        }
      } else if (std::abs(dst[i] - exp[i]) > tol) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void MainTest(const int32_t size) {
    TensorInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeTensorKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid<T>(param.dst, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }

  template <typename T>
  static void CreateNearBoundaryTensorInput(TensorInputParam<T> &param) {
    param.dst = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    const std::vector<T> inputs = {T(0.0f),     T(0.5f),       T(-0.5f),      T(0.9f),    T(-0.9f),
                                   T(0.99f),    T(-0.99f),     T(0.999f),     T(-0.999f), T(0.9999f),
                                   T(-0.9999f), T(0.9998279f), T(-0.9998279f)};
    for (size_t i = 0; i < inputs.size(); ++i) {
      param.src[i] = inputs[i];
      param.exp[i] = static_cast<T>(boost::math::erf_inv(static_cast<double>(inputs[i])));
    }
  }

  template <typename T>
  static void MainNearBoundaryTest(const int32_t size) {
    TensorInputParam<T> param{};
    param.size = size;
    CreateNearBoundaryTensorInput(param);

    auto kernel = [&param] { InvokeTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid<T>(param.dst, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }

  template <typename T>
  static void MainSpecialTest(const int32_t size) {
    TensorInputParam<T> param{};
    param.size = size;
    CreateSpecialTensorInput(param);

    auto kernel = [&param] { InvokeTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid<T>(param.dst, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestApiErfinv, Erfinv_Special_Success) {
  MainSpecialTest<float>(10);
}

TEST_F(TestApiErfinv, Erfinv_Float_256_Success) {
  MainTest<float>(256);
}

TEST_F(TestApiErfinv, Erfinv_NearBoundary_Success) {
  MainNearBoundaryTest<float>(13);
}

}  // namespace af
