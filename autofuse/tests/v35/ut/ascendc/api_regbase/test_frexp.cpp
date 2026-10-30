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
#include <cstdint>
#include <cstdio>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/utils.h"
#include "api_regbase/frexp.h"

using namespace AscendC;

namespace af {

struct FrexpInputParam {
  float *src{};
  float *mantissa{};
  int32_t *exponent{};
  float *mantissa_exp{};
  int32_t *exponent_exp{};
  uint32_t size{};
};

class TestApiFrexp : public testing::Test {
 protected:
  static void InvokeFrexpKernel(FrexpInputParam &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> srcBuf, mantissaBuf, exponentBuf, tmpBuf;
    tpipe.InitBuffer(srcBuf, sizeof(float) * param.size);
    tpipe.InitBuffer(mantissaBuf, sizeof(float) * param.size);
    tpipe.InitBuffer(exponentBuf, sizeof(int32_t) * param.size);
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);

    LocalTensor<float> srcTensor = srcBuf.Get<float>();
    LocalTensor<float> mantissaTensor = mantissaBuf.Get<float>();
    LocalTensor<int32_t> exponentTensor = exponentBuf.Get<int32_t>();
    LocalTensor<uint8_t> tmpTensor = tmpBuf.Get<uint8_t>();

    GmToUb(srcTensor, param.src, param.size);
    FrexpExtend(mantissaTensor, exponentTensor, srcTensor, tmpTensor, param.size);
    UbToGm(param.mantissa, mantissaTensor, param.size);

    for (uint32_t i = 0; i < param.size; i++) {
      param.exponent[i] = exponentTensor.GetValue(i);
    }
  }

  static void AddTestPoint(std::vector<float> &src_list, std::vector<float> &mantissa_exp_list,
                           std::vector<int32_t> &exponent_exp_list, float src, float mant_exp, int32_t exp_exp) {
    src_list.push_back(src);
    mantissa_exp_list.push_back(mant_exp);
    exponent_exp_list.push_back(exp_exp);
  }

  static void AddFrexpTestPoint(std::vector<float> &src_list, std::vector<float> &mantissa_exp_list,
                                std::vector<int32_t> &exponent_exp_list, float src) {
    int exp;
    src_list.push_back(src);
    mantissa_exp_list.push_back(std::frexp(src, &exp));
    exponent_exp_list.push_back(exp);
  }

  static FrexpInputParam BuildFrexpTestData() {
    std::vector<float> src_list, mantissa_exp_list;
    std::vector<int32_t> exponent_exp_list;

    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, 0.0f, 0.0f, 0);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, -0.0f, -0.0f, 0);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, 1.0f, 0.5f, 1);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, 2.0f, 0.5f, 2);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, 3.0f, 0.75f, 2);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, 8.0f, 0.5f, 4);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, 0.5f, 0.5f, 0);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, 0.25f, 0.5f, -1);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, 0.125f, 0.5f, -2);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, -1.0f, -0.5f, 1);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, -2.5f, -0.625f, 2);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, 1024.0f, 0.5f, 11);
    AddFrexpTestPoint(src_list, mantissa_exp_list, exponent_exp_list, 1e-10f);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, INFINITY, INFINITY, 0);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, -INFINITY, -INFINITY, 0);
    AddTestPoint(src_list, mantissa_exp_list, exponent_exp_list, NAN, NAN, 0);

    uint32_t subnormal_bits[] = {0x00800000, 0x00400000, 0x00000001};
    for (uint32_t bits : subnormal_bits) {
      float val = *reinterpret_cast<float *>(&bits);
      AddFrexpTestPoint(src_list, mantissa_exp_list, exponent_exp_list, val);
    }

    FrexpInputParam param{};
    uint32_t n = (uint32_t)src_list.size();
    param.size = n;
    param.src = static_cast<float *>(GmAlloc(sizeof(float) * n));
    param.mantissa = static_cast<float *>(GmAlloc(sizeof(float) * n));
    param.exponent = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * n));
    param.mantissa_exp = static_cast<float *>(GmAlloc(sizeof(float) * n));
    param.exponent_exp = static_cast<int32_t *>(GmAlloc(sizeof(int32_t) * n));

    for (uint32_t i = 0; i < n; ++i) {
      param.src[i] = src_list[i];
      param.mantissa_exp[i] = mantissa_exp_list[i];
      param.exponent_exp[i] = exponent_exp_list[i];
    }
    return param;
  }

  static uint32_t ValidFrexp(float *mantissa, int32_t *exponent, float *mantissa_exp, int32_t *exponent_exp,
                             size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      bool match = false;

      if (std::isnan(mantissa_exp[i])) {
        match = std::isnan(mantissa[i]);
      } else if (std::isinf(mantissa_exp[i])) {
        match = mantissa[i] == mantissa_exp[i];
      } else if (mantissa_exp[i] == 0.0f) {
        match = (mantissa[i] == 0.0f) && (std::signbit(mantissa[i]) == std::signbit(mantissa_exp[i])) &&
                (exponent[i] == exponent_exp[i]);
      } else {
        match = (std::abs(mantissa[i] - mantissa_exp[i]) <= 1e-6) && (exponent[i] == exponent_exp[i]);
      }

      if (!match) {
        diff_count++;
        printf("Test[%u] FAIL: mant_exp=%.9g(e=%d), mant_act=%.9g(e=%d)\n", i, mantissa_exp[i], exponent_exp[i],
               mantissa[i], exponent[i]);
      }
    }
    return diff_count;
  }

  static void FrexpTest() {
    auto param = BuildFrexpTestData();

    auto kernel = [&param] { InvokeFrexpKernel(param); };

    SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count =
        ValidFrexp(param.mantissa, param.exponent, param.mantissa_exp, param.exponent_exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestApiFrexp, frexp_test) {
  FrexpTest();
}

}  // namespace af
