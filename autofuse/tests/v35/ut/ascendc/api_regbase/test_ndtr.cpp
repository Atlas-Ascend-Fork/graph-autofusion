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
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/ndtr.h"

using namespace AscendC;

namespace af {

template <typename T>
struct NdtrInputParam {
  T *dst{};
  T *exp{};
  T *src{};
  int32_t size{0};
};

class TestApiNdtr : public testing::Test {
 protected:
  template <typename T>
  static void InvokeTensorKernel(NdtrInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> srcBuf, dstBuf, tmpBuf;
    tpipe.InitBuffer(srcBuf, sizeof(T) * param.size);
    tpipe.InitBuffer(dstBuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmpBuf, 2048 * sizeof(uint8_t));

    LocalTensor<T> l_src = srcBuf.Get<T>();
    LocalTensor<T> l_dst = dstBuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmpBuf.Get<uint8_t>();

    GmToUb(l_src, param.src, param.size);
    NdtrExtend<T>(l_dst, l_src, l_tmp, param.size);
    UbToGm(param.dst, l_dst, param.size);
  }

  template <typename T>
  static void CreateTensorInput(NdtrInputParam<T> &param) {
    param.dst = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    auto distr = []() {
      if constexpr (std::is_same_v<T, float>) {
        // Use range within reliable Pade approximation: [-3.5, 3.5]
        // Threshold is 3.92, values >= threshold use boundary values
        // So we test Pade approximation in its reliable range
        return std::uniform_real_distribution<float>(-3.5f, 3.5f);
      }
    }();

    constexpr T inv_sqrt2 = 0.7071067811865475f;  // 1/sqrt(2)

    for (int i = 0; i < param.size; i++) {
      T input = static_cast<T>(distr(eng));
      param.src[i] = input;
      // Reference: ndtr(x) = 0.5 * [1 + erf(x/sqrt(2))]
      param.exp[i] = static_cast<T>(0.5 * (1.0 + std::erf(input * inv_sqrt2)));
    }
  }

  template <typename T>
  static void CreateLargeInput(NdtrInputParam<T> &param) {
    param.dst = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    T inf_val = std::numeric_limits<T>::infinity();
    constexpr T inv_sqrt2 = 0.7071067811865475f;

    // Use erfc to preserve finite negative-tail reference values.

    // Case 0: x = 4.0
    param.src[0] = 4.0f;
    param.exp[0] = static_cast<T>(0.5 * std::erfc(-4.0 * inv_sqrt2));

    // Case 1: x = -4.0
    param.src[1] = -4.0f;
    param.exp[1] = static_cast<T>(0.5 * std::erfc(4.0 * inv_sqrt2));

    // Case 2: x = 4.5
    param.src[2] = 4.5f;
    param.exp[2] = static_cast<T>(0.5 * std::erfc(-4.5 * inv_sqrt2));

    // Case 3: x = -4.5
    param.src[3] = -4.5f;
    param.exp[3] = static_cast<T>(0.5 * std::erfc(4.5 * inv_sqrt2));

    // Case 4: x = 6.0 (large positive)
    param.src[4] = 6.0f;
    param.exp[4] = static_cast<T>(0.5 * std::erfc(-6.0 * inv_sqrt2));

    // Case 5: x = -6.0 (large negative)
    param.src[5] = -6.0f;
    param.exp[5] = static_cast<T>(0.5 * std::erfc(6.0 * inv_sqrt2));

    // Case 6: x = +inf
    param.src[6] = inf_val;
    param.exp[6] = 1.0f;

    // Case 7: x = -inf
    param.src[7] = -inf_val;
    param.exp[7] = 0.0f;

    // Case 8: x = 3.93
    param.src[8] = 3.93f;
    param.exp[8] = static_cast<T>(0.5 * std::erfc(-3.93 * inv_sqrt2));

    // Case 9: x = -3.93
    param.src[9] = -3.93f;
    param.exp[9] = static_cast<T>(0.5 * std::erfc(3.93 * inv_sqrt2));
  }

  template <typename T>
  static void CreateSpecialTensorInput(NdtrInputParam<T> &param) {
    param.dst = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    T nan_val = std::numeric_limits<T>::quiet_NaN();
    T inf_val = std::numeric_limits<T>::infinity();

    constexpr T inv_sqrt2 = 0.7071067811865475f;

    // Test case 0: x = 0, ndtr(0) = 0.5
    param.src[0] = 0.0f;
    param.exp[0] = 0.5f;

    // Test case 1: x = 1, ndtr(1) ≈ 0.8413
    param.src[1] = 1.0f;
    param.exp[1] = static_cast<T>(0.5 * (1.0 + std::erf(inv_sqrt2)));

    // Test case 2: x = -1, ndtr(-1) ≈ 0.1587
    param.src[2] = -1.0f;
    param.exp[2] = static_cast<T>(0.5 * (1.0 + std::erf(-inv_sqrt2)));

    // Test case 3: x = +inf, ndtr(+inf) = 1.0
    param.src[3] = inf_val;
    param.exp[3] = 1.0f;

    // Test case 4: x = -inf, ndtr(-inf) = 0.0
    param.src[4] = -inf_val;
    param.exp[4] = 0.0f;

    // Test case 5: x = nan, ndtr(nan) = nan
    param.src[5] = nan_val;
    param.exp[5] = nan_val;

    // Test case 6: x = 2.0, ndtr(2) ≈ 0.9772
    param.src[6] = 2.0f;
    param.exp[6] = static_cast<T>(0.5 * (1.0 + std::erf(2.0 * inv_sqrt2)));

    // Test case 7: x = -2.0, ndtr(-2) ≈ 0.0228
    param.src[7] = -2.0f;
    param.exp[7] = static_cast<T>(0.5 * (1.0 + std::erf(-2.0 * inv_sqrt2)));

    // Test case 8: x = 3.5, still within Pade range (below threshold 3.92)
    param.src[8] = 3.5f;
    param.exp[8] = static_cast<T>(0.5 * (1.0 + std::erf(3.5 * inv_sqrt2)));

    // Test case 9: x = -3.5, still within Pade range
    param.src[9] = -3.5f;
    param.exp[9] = static_cast<T>(0.5 * (1.0 + std::erf(-3.5 * inv_sqrt2)));

    // Test case 10: x = 4.0
    param.src[10] = 4.0f;
    param.exp[10] = static_cast<T>(0.5 * std::erfc(-4.0 * inv_sqrt2));

    // Test case 11: x = -4.0
    param.src[11] = -4.0f;
    param.exp[11] = static_cast<T>(0.5 * std::erfc(4.0 * inv_sqrt2));
  }

  template <typename T>
  static uint32_t Valid(T *dst, T *exp, T *src, size_t comp_size) {
    uint32_t diff_count = 0;

    for (uint32_t i = 0; i < comp_size; i++) {
      bool is_diff = false;
      T abs_diff = 0;
      T rel_err = 0;

      if (std::isnan(exp[i])) {
        if (!std::isnan(dst[i])) {
          is_diff = true;
        }
      } else if (std::isinf(exp[i])) {
        if (dst[i] != exp[i]) {
          is_diff = true;
        }
      } else {
        abs_diff = std::abs(dst[i] - exp[i]);

        // Safe division: handle multiple cases to prevent division by zero
        T abs_exp = std::abs(exp[i]);

        if (abs_exp < T(1e-10)) {
          // When exp[i] is very close to 0 or exactly 0
          // Use absolute error instead of relative error
          // For boundary values (0.0 or 1.0), use tolerance of 1e-5
          if (abs_diff > T(1e-5)) {
            rel_err = T(1.0);  // Mark as large error to trigger failure
            is_diff = true;
          } else {
            rel_err = 0;  // Accept small absolute difference for boundary values
          }
        } else {
          // Normal case: exp[i] is not close to 0
          // Compute relative error with safe denominator
          T rel_den = std::max(abs_exp, T(1e-10));
          rel_err = abs_diff / rel_den;

          // Use relative error threshold of 1e-3 for Pade approximation
          // Some precision loss is expected for polynomial approximation
          if (rel_err > T(1e-3)) {
            is_diff = true;
          }
        }
      }

      if (is_diff) {
        diff_count++;
      }
    }

    return diff_count;
  }

  template <typename T>
  static void MainTest(const int32_t size) {
    NdtrInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    auto kernel = [&param] { InvokeTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid<T>(param.dst, param.exp, param.src, param.size);
    EXPECT_EQ(diff_count, 0);

    AscendC::GmFree(param.dst);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src);
  }

  template <typename T>
  static void MainSpecialTest(const int32_t size) {
    NdtrInputParam<T> param{};
    param.size = size;
    CreateSpecialTensorInput(param);

    auto kernel = [&param] { InvokeTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid<T>(param.dst, param.exp, param.src, param.size);
    EXPECT_EQ(diff_count, 0);

    AscendC::GmFree(param.dst);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src);
  }

  template <typename T>
  static void MainLargeTest(const int32_t size) {
    NdtrInputParam<T> param{};
    param.size = size;
    CreateLargeInput(param);

    auto kernel = [&param] { InvokeTensorKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid<T>(param.dst, param.exp, param.src, param.size);
    EXPECT_EQ(diff_count, 0);

    AscendC::GmFree(param.dst);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.src);
  }
};

TEST_F(TestApiNdtr, Ndtr_Special_Success) {
  MainSpecialTest<float>(12);
}

TEST_F(TestApiNdtr, Ndtr_Large_Success) {
  MainLargeTest<float>(10);
}

TEST_F(TestApiNdtr, Ndtr_Float_256_Success) {
  MainTest<float>(256);
}

TEST_F(TestApiNdtr, Ndtr_Float_1024_Success) {
  MainTest<float>(1024);
}

}  // namespace af
