/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. This file is provided on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/sigmoid.h"

using namespace AscendC;

namespace af {
namespace {

// 覆盖正侧饱和区（x ≥ 88 时 e^-x 落入 fp32 次正规，--cce-ftz=false 下 Reg::Exp 误返回 ~1）
// 与负侧次正规区（x = -88 时 golden ≈ 6e-39 为次正规值）。
constexpr std::array<float, 21> kSigmoidInputs = {
    -20.0F, -10.0F, -5.0F,  -1.0F, -0.5F, -0.0F, 0.0F,  0.5F,  1.0F,   5.0F,   10.0F,
    20.0F,  80.0F,  -80.0F, 0.1F,  -0.1F, 2.0F,  -2.0F, 88.0F, -88.0F, 100.0F,
};

template <typename T>
struct SigmoidInputParam {
  T *src{};
  T *dst{};
  T *expect{};
  uint32_t size{};
};

template <typename T>
double SigmoidReference(T input) {
  const double x = static_cast<double>(input);
  if (x >= 0.0) {
    return 1.0 / (1.0 + std::exp(-x));
  }
  const double exp_x = std::exp(x);
  return exp_x / (1.0 + exp_x);
}

template <typename T>
double GetTolerance() {
  if constexpr (std::is_same_v<T, half>) {
    return 1e-3;
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    return 1e-2;
  }
  return 1e-5;
}

template <typename T>
bool IsClose(T actual, T expect) {
  const double actual_value = static_cast<double>(actual);
  const double expect_value = static_cast<double>(expect);
  if (std::isnan(expect_value)) {
    return std::isnan(actual_value);
  }
  if (std::isinf(expect_value)) {
    return std::isinf(actual_value) && std::signbit(actual_value) == std::signbit(expect_value);
  }
  const double tolerance = GetTolerance<T>();
  return std::abs(actual_value - expect_value) <= tolerance + tolerance * std::abs(expect_value);
}

template <typename T>
void InvokeSigmoid(SigmoidInputParam<T> &param, bool inplace) {
  TPipe tpipe;
  TBuf<TPosition::VECCALC> src_buf, dst_buf, tmp_buf;
  const uint32_t aligned_size = AlignUp(param.size, ONE_BLK_SIZE / sizeof(T));
  tpipe.InitBuffer(src_buf, sizeof(T) * aligned_size);
  if (!inplace) {
    tpipe.InitBuffer(dst_buf, sizeof(T) * aligned_size);
  }
  tpipe.InitBuffer(tmp_buf, TMP_UB_SIZE);

  LocalTensor<T> src = src_buf.Get<T>();
  LocalTensor<T> dst = inplace ? src : dst_buf.Get<T>();
  LocalTensor<uint8_t> tmp = tmp_buf.Get<uint8_t>();

  GmToUb(src, param.src, param.size);
  SigmoidExtend(dst, src, tmp, param.size);
  UbToGm(param.dst, dst, param.size);
}

template <typename T>
void CreateInput(SigmoidInputParam<T> &param) {
  param.src = static_cast<T *>(GmAlloc(sizeof(T) * param.size));
  param.dst = static_cast<T *>(GmAlloc(sizeof(T) * param.size));
  param.expect = static_cast<T *>(GmAlloc(sizeof(T) * param.size));

  for (uint32_t i = 0; i < param.size; ++i) {
    param.src[i] = static_cast<T>(kSigmoidInputs[i % kSigmoidInputs.size()]);
    param.expect[i] = static_cast<T>(SigmoidReference(param.src[i]));
  }
}

template <typename T>
void CreateSpecialInput(SigmoidInputParam<T> &param) {
  param.src = static_cast<T *>(GmAlloc(sizeof(T) * param.size));
  param.dst = static_cast<T *>(GmAlloc(sizeof(T) * param.size));
  param.expect = static_cast<T *>(GmAlloc(sizeof(T) * param.size));

  const T nan_value = std::numeric_limits<T>::quiet_NaN();
  const T inf_value = std::numeric_limits<T>::infinity();
  const std::array<T, 9> inputs = {T(0.0),   T(-0.0),   T(1.0),     T(-1.0),  T(20.0),
                                   T(-20.0), inf_value, -inf_value, nan_value};
  for (uint32_t i = 0; i < param.size; ++i) {
    param.src[i] = inputs[i % inputs.size()];
    param.expect[i] = static_cast<T>(SigmoidReference(param.src[i]));
  }
}

template <typename T>
void FreeInput(SigmoidInputParam<T> &param) {
  GmFree(param.src);
  GmFree(param.dst);
  GmFree(param.expect);
}

template <typename T>
void ValidateOutput(const SigmoidInputParam<T> &param) {
  uint32_t mismatch_count = 0;
  for (uint32_t i = 0; i < param.size; ++i) {
    if (!IsClose(param.dst[i], param.expect[i])) {
      ++mismatch_count;
      printf("diff at index %u: src=%f, actual=%f, expect=%f\n", i, static_cast<float>(param.src[i]),
             static_cast<float>(param.dst[i]), static_cast<float>(param.expect[i]));
    }
  }
  EXPECT_EQ(mismatch_count, 0U) << " of " << param.size;
}

template <typename T>
void RunSigmoidTest(uint32_t size, bool inplace, bool special_input = false) {
  SigmoidInputParam<T> param{};
  param.size = size;
  if (special_input) {
    CreateSpecialInput(param);
  } else {
    CreateInput(param);
  }

  auto kernel = [&param, inplace] { InvokeSigmoid(param, inplace); };
  SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);

  ValidateOutput(param);
  FreeInput(param);
}

template <typename T>
void RunSigmoidSizes(bool inplace) {
  RunSigmoidTest<T>((ONE_BLK_SIZE - sizeof(T)) / sizeof(T), inplace);
  RunSigmoidTest<T>(ONE_REPEAT_BYTE_SIZE / sizeof(T), inplace);
  RunSigmoidTest<T>((ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE - sizeof(T)) / sizeof(T), inplace);
  RunSigmoidTest<T>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(T), inplace);
}

}  // namespace

class TestRegbaseApiSigmoidUT : public testing::Test {};

TEST_F(TestRegbaseApiSigmoidUT, Sigmoid_TensorTensor_Half) {
  RunSigmoidSizes<half>(false);
}

TEST_F(TestRegbaseApiSigmoidUT, Sigmoid_TensorTensor_Float) {
  RunSigmoidSizes<float>(false);
}

TEST_F(TestRegbaseApiSigmoidUT, Sigmoid_TensorTensor_Bfloat16) {
  RunSigmoidSizes<bfloat16_t>(false);
}

TEST_F(TestRegbaseApiSigmoidUT, Sigmoid_Inplace_Half) {
  RunSigmoidSizes<half>(true);
}

TEST_F(TestRegbaseApiSigmoidUT, Sigmoid_Inplace_Float) {
  RunSigmoidSizes<float>(true);
}

TEST_F(TestRegbaseApiSigmoidUT, Sigmoid_Inplace_Bfloat16) {
  RunSigmoidSizes<bfloat16_t>(true);
}

TEST_F(TestRegbaseApiSigmoidUT, Sigmoid_Special_Float) {
  RunSigmoidTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float), false, true);
}

}  // namespace af
