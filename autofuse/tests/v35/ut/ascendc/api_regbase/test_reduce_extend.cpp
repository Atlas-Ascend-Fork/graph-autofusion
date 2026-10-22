/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/reduce_extend.h"

namespace {
constexpr uint32_t kTmpBufferSize = 32 * 1024;
constexpr uint32_t kReduceMinRaInputCount = 32;
constexpr uint32_t kReduceMinRaOutputCount = 8;
constexpr uint32_t kReduceMinRaStorageCount = 80;

template <typename T, typename Pattern, typename ReduceFunc>
void RunReduceKernel(T *input, T *output, uint32_t inputCount, uint32_t outputCount, uint32_t dim0, uint32_t dim1,
                     ReduceFunc reduce) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> srcBuffer;
  TBuf<TPosition::VECCALC> dstBuffer;
  TBuf<TPosition::VECCALC> tmpBuffer;
  pipe.InitBuffer(srcBuffer, AlignUp(inputCount * sizeof(T), ONE_BLK_SIZE));
  pipe.InitBuffer(dstBuffer, AlignUp(outputCount * sizeof(T), ONE_BLK_SIZE));
  pipe.InitBuffer(tmpBuffer, kTmpBufferSize);

  LocalTensor<T> src = srcBuffer.Get<T>();
  LocalTensor<T> dst = dstBuffer.Get<T>();
  LocalTensor<uint8_t> tmp = tmpBuffer.Get<uint8_t>();
  GmToUb(src, input, inputCount);

  const uint32_t shape[] = {dim0, dim1};
  reduce(dst, src, tmp, shape);
  UbToGm(output, dst, outputCount);
}

template <typename T, typename Pattern, typename ReduceFunc>
std::vector<T> RunReduce(const std::vector<T> &input, uint32_t outputCount, uint32_t dim0, uint32_t dim1,
                         ReduceFunc reduce) {
  const uint32_t inputBytes = AlignUp(input.size() * sizeof(T), ONE_BLK_SIZE);
  const uint32_t outputBytes = AlignUp(outputCount * sizeof(T), ONE_BLK_SIZE);
  auto *inputGm = static_cast<T *>(AscendC::GmAlloc(inputBytes));
  auto *outputGm = static_cast<T *>(AscendC::GmAlloc(outputBytes));
  std::copy(input.begin(), input.end(), inputGm);
  std::fill_n(outputGm, outputBytes / sizeof(T), T{});

  auto kernel = [reduce](T *kernelInput, T *kernelOutput, uint32_t inputSize, uint32_t outputSize, uint32_t shape0,
                         uint32_t shape1) {
    RunReduceKernel<T, Pattern>(kernelInput, kernelOutput, inputSize, outputSize, shape0, shape1, reduce);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, inputGm, outputGm, input.size(), outputCount, dim0, dim1);

  std::vector<T> output(outputGm, outputGm + outputCount);
  AscendC::GmFree(inputGm);
  AscendC::GmFree(outputGm);
  return output;
}

void RunReduceMinRaReuseSourceKernel(float *input, float *output, float *sourceAfter) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> srcBuffer;
  TBuf<TPosition::VECCALC> dstBuffer;
  pipe.InitBuffer(srcBuffer, kReduceMinRaStorageCount * sizeof(float));
  pipe.InitBuffer(dstBuffer, kReduceMinRaOutputCount * sizeof(float));

  LocalTensor<float> src = srcBuffer.Get<float>();
  LocalTensor<float> dst = dstBuffer.Get<float>();
  GmToUb(src, input, kReduceMinRaStorageCount);

  const uint32_t shape[] = {4, 8};
  AscendC::ReduceMinExtend<float, AscendC::Pattern::Reduce::RA, true>(dst, src, shape, true);
  UbToGm(output, dst, kReduceMinRaOutputCount);
  UbToGm(sourceAfter, src, kReduceMinRaStorageCount);
}
}  // namespace

TEST(TestApiReduceExtend, ReduceSumFloatArAndRa) {
  std::vector<float> arInput(16);
  std::iota(arInput.begin(), arInput.end(), 1.0F);
  auto arOutput = RunReduce<float, AscendC::Pattern::Reduce::AR>(
      arInput, 2, 2, 8, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceSumExtend<float, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, false);
      });
  EXPECT_EQ(arOutput, (std::vector<float>{36.0F, 100.0F}));

  std::vector<float> raInput(32);
  std::iota(raInput.begin(), raInput.end(), 1.0F);
  auto raOutput = RunReduce<float, AscendC::Pattern::Reduce::RA>(
      raInput, 8, 4, 8, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceSumExtend<float, AscendC::Pattern::Reduce::RA>(dst, src, tmp, shape, false);
      });
  for (uint32_t i = 0; i < raOutput.size(); ++i) {
    EXPECT_FLOAT_EQ(raOutput[i], 52.0F + 4.0F * i);
  }
}

TEST(TestApiReduceExtend, ReduceSumSupportsInt8AndInt16) {
  std::vector<int8_t> int8Input(64, 1);
  auto int8Output = RunReduce<int8_t, AscendC::Pattern::Reduce::AR>(
      int8Input, 2, 2, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceSumExtend<int8_t, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, false);
      });
  EXPECT_EQ(int8Output, (std::vector<int8_t>{32, 32}));

  std::vector<int16_t> int16Input(32, 2);
  auto int16Output = RunReduce<int16_t, AscendC::Pattern::Reduce::AR>(
      int16Input, 2, 2, 16, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceSumExtend<int16_t, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, false);
      });
  EXPECT_EQ(int16Output, (std::vector<int16_t>{32, 32}));
}

TEST(TestApiReduceExtend, ArithmeticAndLogicalReduceEntries) {
  const std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8};

  auto maxOutput = RunReduce<float, AscendC::Pattern::Reduce::AR>(
      input, 2, 2, 8, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceMaxExtend<float, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, false);
      });
  EXPECT_EQ(maxOutput, (std::vector<float>{8.0F, -1.0F}));

  auto minOutput = RunReduce<float, AscendC::Pattern::Reduce::AR>(
      input, 2, 2, 8, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceMinExtend<float, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, false);
      });
  EXPECT_EQ(minOutput, (std::vector<float>{1.0F, -8.0F}));

  auto meanOutput = RunReduce<float, AscendC::Pattern::Reduce::AR>(
      input, 2, 2, 8, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceMeanExtend<float, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
      });
  EXPECT_EQ(meanOutput, (std::vector<float>{4.5F, -4.5F}));

  const std::vector<float> prodInput(16, 1.0F);
  auto prodOutput = RunReduce<float, AscendC::Pattern::Reduce::AR>(
      prodInput, 2, 2, 8, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<float, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
      });
  EXPECT_EQ(prodOutput, (std::vector<float>{1.0F, 1.0F}));

  std::vector<uint8_t> logicalInput(64, 1);
  logicalInput[7] = 0;
  logicalInput[32] = 0;
  auto anyOutput = RunReduce<uint8_t, AscendC::Pattern::Reduce::AR>(
      logicalInput, 2, 2, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceAnyExtend<uint8_t, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
      });
  EXPECT_EQ(anyOutput, (std::vector<uint8_t>{1, 1}));

  auto allOutput = RunReduce<uint8_t, AscendC::Pattern::Reduce::AR>(
      logicalInput, 2, 2, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceAllExtend<uint8_t, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
      });
  EXPECT_EQ(allOutput, (std::vector<uint8_t>{0, 0}));
}

TEST(TestApiReduceExtend, ReduceMinRaReuseSourceDoesNotWriteBeyondInput) {
  std::vector<float> input(kReduceMinRaStorageCount);
  std::iota(input.begin(), input.begin() + kReduceMinRaInputCount, 1.0F);
  for (uint32_t i = kReduceMinRaInputCount; i < kReduceMinRaStorageCount; ++i) {
    input[i] = -static_cast<float>(i);
  }

  const uint32_t inputBytes = input.size() * sizeof(float);
  const uint32_t outputBytes = kReduceMinRaOutputCount * sizeof(float);
  auto *inputGm = static_cast<float *>(AscendC::GmAlloc(inputBytes));
  auto *outputGm = static_cast<float *>(AscendC::GmAlloc(outputBytes));
  auto *sourceAfterGm = static_cast<float *>(AscendC::GmAlloc(inputBytes));
  std::copy(input.begin(), input.end(), inputGm);
  std::fill_n(outputGm, kReduceMinRaOutputCount, 0.0F);
  std::fill_n(sourceAfterGm, kReduceMinRaStorageCount, 0.0F);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(RunReduceMinRaReuseSourceKernel, 1, inputGm, outputGm, sourceAfterGm);

  for (uint32_t i = 0; i < kReduceMinRaOutputCount; ++i) {
    EXPECT_FLOAT_EQ(outputGm[i], static_cast<float>(i + 1));
  }
  for (uint32_t i = kReduceMinRaInputCount; i < kReduceMinRaStorageCount; ++i) {
    EXPECT_FLOAT_EQ(sourceAfterGm[i], input[i]) << "source guard changed at index " << i;
  }

  AscendC::GmFree(inputGm);
  AscendC::GmFree(outputGm);
  AscendC::GmFree(sourceAfterGm);
}
