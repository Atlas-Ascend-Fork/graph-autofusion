/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
using namespace AscendC;

#include "utils.h"
#include "brc_inline_api.h"

enum class BinaryOp { Add, Sub, Div, Mul };

constexpr int64_t kLargeFirstAxisStride = 2048;
constexpr float kCounterPathMarker = 1.0F;
constexpr float kRepeatPathMarker = -1.0F;

inline __aicore__ void MarkCounterPath(const LocalTensor<float> &dst, const LocalTensor<float> &,
                                       const LocalTensor<float> &, const int32_t &) {
  dst.SetValue(0, kCounterPathMarker);
}

inline __aicore__ void MarkRepeatPath(const LocalTensor<float> &dst, const LocalTensor<float> &,
                                      const LocalTensor<float> &, uint64_t, const uint8_t, const BinaryRepeatParams &) {
  dst.SetValue(0, kRepeatPathMarker);
}

template <class T>
void UbToGm(T *gm, LocalTensor<T> &local, uint64_t size) {
  for (int i = 0; i < size; i++) {
    gm[i] = local.GetValue(i);
  }
}

template <class T>
void GmToUb(LocalTensor<T> &local, T *gm, int size) {
  for (int i = 0; i < size; i++) {
    local.SetValue(i, gm[i]);
  }
}

template <class T>
void TestBrcInlineTwoDimApi(std::vector<int64_t> &input1_shape, std::vector<int64_t> &input2_shape,
                            std::vector<int64_t> &output_shape, const uint8_t is_input0_block_brc,
                            const uint8_t is_input1_block_brc, BinaryOp op,
                            void (*FUNC1)(const LocalTensor<T> &, const LocalTensor<T> &, const LocalTensor<T> &,
                                          const int32_t &),
                            void (*FUNC2)(const LocalTensor<T> &, const LocalTensor<T> &, const LocalTensor<T> &,
                                          uint64_t, const uint8_t, const BinaryRepeatParams &)) {
  int input1_size, input2_size, output_size;
  int first_axis_v_stride = (output_shape[1] * sizeof(T) + 31) / 32 * 32 / sizeof(T);
  if (is_input1_block_brc) {
    input1_size = (input1_shape[0] * sizeof(T) + 31) / 32 * 32 / sizeof(T);
    input2_size = input2_shape[0] * (input2_shape[1] * sizeof(T) + 31) / 32 * 32 / sizeof(T);
    output_size = input2_size;
  } else {
    input1_size = input1_shape[0] * (input1_shape[1] * sizeof(T) + 31) / 32 * 32 / sizeof(T);
    input2_size = (input2_shape[0] * sizeof(T) + 31) / 32 * 32 / sizeof(T);
    output_size = input1_size;
  }

  T *x1 = (T *)AscendC::GmAlloc(sizeof(T) * input1_size);
  T *x2 = (T *)AscendC::GmAlloc(sizeof(T) * input2_size);
  T *y = (T *)AscendC::GmAlloc(sizeof(T) * output_size);
  T *expect = (T *)AscendC::GmAlloc(sizeof(T) * output_size);

  for (int i = 0; i < input1_size; ++i) {
    x1[i] = (T)4;
  }

  for (int i = 0; i < input2_size; ++i) {
    x2[i] = (T)2;
  }

  switch (op) {
    case BinaryOp::Add:
      for (int i = 0; i < output_size; ++i) {
        expect[i] = (T)6;
      }
      break;
    case BinaryOp::Sub:
      for (int i = 0; i < output_size; ++i) {
        expect[i] = (T)2;
      }
      break;
    case BinaryOp::Mul:
      for (int i = 0; i < output_size; ++i) {
        expect[i] = (T)8;
      }
      break;
    case BinaryOp::Div:
      for (int i = 0; i < output_size; ++i) {
        expect[i] = (T)2;
      }
      break;
  }

  // 构造Api调用函数
  auto kernel =
      [](const std::vector<int64_t> &shape, const uint8_t is_input0_block_brc, const uint8_t is_input1_block_brc,
         const int64_t first_axis_v_stride, const int64_t dtype_size,
         void (*FUNC1)(const LocalTensor<T> &, const LocalTensor<T> &, const LocalTensor<T> &, const int32_t &),
         void (*FUNC2)(const LocalTensor<T> &, const LocalTensor<T> &, const LocalTensor<T> &, uint64_t, const uint8_t,
                       const BinaryRepeatParams &),
         T *x1, T *x2, T *y) {
        TPipe pipe;
        TBuf<TPosition::VECIN> x1Tensor, x2Tensor, outTensor, tmpTensor;
        int shape_size = shape[0] * first_axis_v_stride;
        int64_t tmpSize = shape_size * sizeof(T);

        if (is_input0_block_brc && !is_input1_block_brc) {
          pipe.InitBuffer(x1Tensor, tmpSize);
          pipe.InitBuffer(x2Tensor, first_axis_v_stride * sizeof(T));
        } else {
          pipe.InitBuffer(x1Tensor, first_axis_v_stride * sizeof(T));
          pipe.InitBuffer(x2Tensor, tmpSize);
        }

        pipe.InitBuffer(tmpTensor, tmpSize);
        pipe.InitBuffer(outTensor, tmpSize);
        LocalTensor<uint8_t> tmp_buf = tmpTensor.Get<uint8_t>();
        LocalTensor<T> input1 = x1Tensor.Get<T>();
        LocalTensor<T> input2 = x2Tensor.Get<T>();
        LocalTensor<T> output = outTensor.Get<T>();

        if (is_input0_block_brc && !is_input1_block_brc) {
          GmToUb(input1, x1, shape_size);
          GmToUb(input2, x2, first_axis_v_stride);
        } else {
          GmToUb(input1, x1, first_axis_v_stride);
          GmToUb(input2, x2, shape_size);
        }

        BinaryBrcInlineApiWithTwoVectorizedAxis<T>(output, input1, input2, shape[0], shape[1], is_input0_block_brc,
                                                   is_input1_block_brc, first_axis_v_stride, dtype_size, FUNC1, FUNC2);
        UbToGm(y, output, shape_size);
      };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  ICPU_RUN_KF(kernel, 1, output_shape, is_input0_block_brc, is_input1_block_brc, first_axis_v_stride, sizeof(T), FUNC1,
              FUNC2, x1, x2, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < output_shape[0]; ++i) {
    for (int j = 0; j < output_shape[1]; ++j) {
      auto diff = (double)(y[i * first_axis_v_stride + j] - expect[i * first_axis_v_stride + j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }
  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiBroInline, TestLargeRepeatStrideFallsBackToCounterPath) {
  constexpr int64_t kOuterSize = 2;
  constexpr int64_t kInnerSize = 64;
  constexpr int64_t kBufferSize = kOuterSize * kLargeFirstAxisStride;
  constexpr int64_t kOutputSize = 2;
  auto *output = static_cast<float *>(AscendC::GmAlloc(kOutputSize * sizeof(float)));

  auto kernel = [](float *output) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> output_buf;
    pipe.InitBuffer(output_buf, kBufferSize * sizeof(float));
    LocalTensor<float> local_output = output_buf.Get<float>();
    local_output.SetValue(0, 0.0F);
    local_output.SetValue(kLargeFirstAxisStride, 0.0F);
    BinaryBrcInlineApiWithTwoVectorizedAxis<float>(local_output, local_output, local_output, kOuterSize, kInnerSize, 0,
                                                   0, kLargeFirstAxisStride, sizeof(float), &MarkCounterPath,
                                                   &MarkRepeatPath);
    output[0] = local_output.GetValue(0);
    output[1] = local_output.GetValue(kLargeFirstAxisStride);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, output);

  EXPECT_EQ(output[0], kCounterPathMarker);
  EXPECT_EQ(output[1], kCounterPathMarker);
  AscendC::GmFree(output);
}

TEST(TestApiBroInline, Test_Add_Float_Two_Axis) {
  using AddCounterPtr =
      void (*)(const LocalTensor<float> &, const LocalTensor<float> &, const LocalTensor<float> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Add;

  using AddNormalPtr = void (*)(const LocalTensor<float> &, const LocalTensor<float> &, const LocalTensor<float> &,
                                uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Add;
  BinaryOp op = BinaryOp::Add;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},  {192},
                                                    {192},     {200},    {200},     {400, 16}, {16}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},    {192},     {200},    {200}, {58, 192},
                                                    {2, 192}, {40, 200}, {2, 200}, {16},  {400, 16}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192},  {40, 200}, {2, 200},  {58, 192},
                                                    {2, 192},  {40, 200}, {2, 200},  {400, 16}, {400, 16}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0, 1, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1, 0, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<float>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                  is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Add_Half_Two_Axis) {
  using AddCounterPtr =
      void (*)(const LocalTensor<half> &, const LocalTensor<half> &, const LocalTensor<half> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Add;

  using AddNormalPtr = void (*)(const LocalTensor<half> &, const LocalTensor<half> &, const LocalTensor<half> &,
                                uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Add;
  BinaryOp op = BinaryOp::Add;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<half>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                 is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Add_Int32_Two_Axis) {
  using AddCounterPtr = void (*)(const LocalTensor<int32_t> &, const LocalTensor<int32_t> &,
                                 const LocalTensor<int32_t> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Add;

  using AddNormalPtr = void (*)(const LocalTensor<int32_t> &, const LocalTensor<int32_t> &,
                                const LocalTensor<int32_t> &, uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Add;
  BinaryOp op = BinaryOp::Add;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<int32_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                    is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Add_Int16_Two_Axis) {
  using AddCounterPtr = void (*)(const LocalTensor<int16_t> &, const LocalTensor<int16_t> &,
                                 const LocalTensor<int16_t> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Add;

  using AddNormalPtr = void (*)(const LocalTensor<int16_t> &, const LocalTensor<int16_t> &,
                                const LocalTensor<int16_t> &, uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Add;
  BinaryOp op = BinaryOp::Add;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<int16_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                    is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Sub_Float_Two_Axis) {
  using AddCounterPtr =
      void (*)(const LocalTensor<float> &, const LocalTensor<float> &, const LocalTensor<float> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Sub;

  using AddNormalPtr = void (*)(const LocalTensor<float> &, const LocalTensor<float> &, const LocalTensor<float> &,
                                uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Sub;
  BinaryOp op = BinaryOp::Sub;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<float>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                  is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Sub_Half_Two_Axis) {
  using AddCounterPtr =
      void (*)(const LocalTensor<half> &, const LocalTensor<half> &, const LocalTensor<half> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Sub;

  using AddNormalPtr = void (*)(const LocalTensor<half> &, const LocalTensor<half> &, const LocalTensor<half> &,
                                uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Sub;
  BinaryOp op = BinaryOp::Sub;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<half>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                 is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Sub_Int32_Two_Axis) {
  using AddCounterPtr = void (*)(const LocalTensor<int32_t> &, const LocalTensor<int32_t> &,
                                 const LocalTensor<int32_t> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Sub;

  using AddNormalPtr = void (*)(const LocalTensor<int32_t> &, const LocalTensor<int32_t> &,
                                const LocalTensor<int32_t> &, uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Sub;
  BinaryOp op = BinaryOp::Sub;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<int32_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                    is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Sub_Int16_Two_Axis) {
  using AddCounterPtr = void (*)(const LocalTensor<int16_t> &, const LocalTensor<int16_t> &,
                                 const LocalTensor<int16_t> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Sub;

  using AddNormalPtr = void (*)(const LocalTensor<int16_t> &, const LocalTensor<int16_t> &,
                                const LocalTensor<int16_t> &, uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Sub;
  BinaryOp op = BinaryOp::Sub;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<int16_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                    is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Mul_Float_Two_Axis) {
  using AddCounterPtr =
      void (*)(const LocalTensor<float> &, const LocalTensor<float> &, const LocalTensor<float> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Mul;

  using AddNormalPtr = void (*)(const LocalTensor<float> &, const LocalTensor<float> &, const LocalTensor<float> &,
                                uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Mul;
  BinaryOp op = BinaryOp::Mul;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<float>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                  is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Mul_Half_Two_Axis) {
  using AddCounterPtr =
      void (*)(const LocalTensor<half> &, const LocalTensor<half> &, const LocalTensor<half> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Mul;

  using AddNormalPtr = void (*)(const LocalTensor<half> &, const LocalTensor<half> &, const LocalTensor<half> &,
                                uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Mul;
  BinaryOp op = BinaryOp::Mul;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<half>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                 is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Mul_Int32_Two_Axis) {
  using AddCounterPtr = void (*)(const LocalTensor<int32_t> &, const LocalTensor<int32_t> &,
                                 const LocalTensor<int32_t> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Mul;

  using AddNormalPtr = void (*)(const LocalTensor<int32_t> &, const LocalTensor<int32_t> &,
                                const LocalTensor<int32_t> &, uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Mul;
  BinaryOp op = BinaryOp::Mul;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<int32_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                    is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Mul_Int16_Two_Axis) {
  using AddCounterPtr = void (*)(const LocalTensor<int16_t> &, const LocalTensor<int16_t> &,
                                 const LocalTensor<int16_t> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Mul;

  using AddNormalPtr = void (*)(const LocalTensor<int16_t> &, const LocalTensor<int16_t> &,
                                const LocalTensor<int16_t> &, uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Mul;
  BinaryOp op = BinaryOp::Mul;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<int16_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                    is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Div_Float_Two_Axis) {
  using AddCounterPtr =
      void (*)(const LocalTensor<float> &, const LocalTensor<float> &, const LocalTensor<float> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Div;

  using AddNormalPtr = void (*)(const LocalTensor<float> &, const LocalTensor<float> &, const LocalTensor<float> &,
                                uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Div;
  BinaryOp op = BinaryOp::Div;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<float>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                  is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}

TEST(TestApiBroInline, Test_Div_Half_Two_Axis) {
  using AddCounterPtr =
      void (*)(const LocalTensor<half> &, const LocalTensor<half> &, const LocalTensor<half> &, const int32_t &);
  AddCounterPtr cPtr = &AscendC::Div;

  using AddNormalPtr = void (*)(const LocalTensor<half> &, const LocalTensor<half> &, const LocalTensor<half> &,
                                uint64_t, const uint8_t, const BinaryRepeatParams &);
  AddNormalPtr nPtr = &AscendC::Div;
  BinaryOp op = BinaryOp::Div;
  std::vector<std::vector<int64_t>> input1_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {192},     {192},    {200},     {200}};
  std::vector<std::vector<int64_t>> input2_shape = {{192},     {192},    {200},     {200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<std::vector<int64_t>> output_shape = {{58, 192}, {2, 192}, {40, 200}, {2, 200},
                                                    {58, 192}, {2, 192}, {40, 200}, {2, 200}};
  std::vector<uint8_t> is_input1_block_brc = {1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<uint8_t> is_input2_block_brc = {0, 0, 0, 0, 1, 1, 1, 1};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestBrcInlineTwoDimApi<half>(input1_shape[idx], input2_shape[idx], output_shape[idx], is_input1_block_brc[idx],
                                 is_input2_block_brc[idx], op, cPtr, nPtr);
  }
}
