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
#include "api_regbase/cast.h"
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

template <typename InT, typename OutT>
std::vector<OutT> RunCast(const std::vector<InT> &input) {
  const uint32_t inputBytes = AlignUp(input.size() * sizeof(InT), ONE_BLK_SIZE);
  const uint32_t outputBytes = AlignUp(input.size() * sizeof(OutT), ONE_BLK_SIZE);
  auto *inputGm = static_cast<InT *>(AscendC::GmAlloc(inputBytes));
  auto *outputGm = static_cast<OutT *>(AscendC::GmAlloc(outputBytes));
  std::copy(input.begin(), input.end(), inputGm);
  std::fill_n(outputGm, outputBytes / sizeof(OutT), OutT{});

  auto kernel = [](InT *kernelInput, OutT *kernelOutput, uint32_t size) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuffer;
    TBuf<TPosition::VECCALC> dstBuffer;
    pipe.InitBuffer(srcBuffer, AlignUp(size * sizeof(InT), ONE_BLK_SIZE));
    pipe.InitBuffer(dstBuffer, AlignUp(size * sizeof(OutT), ONE_BLK_SIZE));
    LocalTensor<InT> src = srcBuffer.Get<InT>();
    LocalTensor<OutT> dst = dstBuffer.Get<OutT>();
    GmToUb(src, kernelInput, size);
    const uint32_t outputDims[] = {1, size};
    const uint32_t strides[] = {0, 1};
    CastExtend<InT, OutT>(dst, src, outputDims, strides, strides);
    UbToGm(kernelOutput, dst, size);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, inputGm, outputGm, input.size());

  std::vector<OutT> output(outputGm, outputGm + input.size());
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

template <typename T, typename U = T>
void TestProdIntegerDtype(T one, T two) {
  std::vector<U> input(64, static_cast<U>(one));
  input[0] = static_cast<U>(two);
  input[1] = static_cast<U>(two);
  auto output = RunReduce<U, AscendC::Pattern::Reduce::AR>(
      input, 2, 2, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<U, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
      });
  std::vector<U> expected(2, static_cast<U>(one));
  expected[0] = static_cast<U>(two) * static_cast<U>(two);
  EXPECT_EQ(output, expected);

  auto raOutput = RunReduce<U, AscendC::Pattern::Reduce::RA>(
      input, 32, 2, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<U, AscendC::Pattern::Reduce::RA>(dst, src, tmp, shape, true);
      });
  std::vector<U> raExpected(32, static_cast<U>(one));
  raExpected[0] = static_cast<U>(two);
  raExpected[1] = static_cast<U>(two);
  EXPECT_EQ(raOutput, raExpected);
}

TEST(TestApiReduceExtend, ReduceProdSupportsIntegerDtypes) {
  TestProdIntegerDtype<int16_t>(int16_t{1}, int16_t{2});
  TestProdIntegerDtype<int32_t>(int32_t{1}, int32_t{2});
  TestProdIntegerDtype<int64_t>(int64_t{1}, int64_t{2});
}

TEST(TestApiReduceExtend, ReduceProdInt32ArNoReuseSourceCoversAlignedTail) {
  constexpr uint32_t kDimA = 2;
  constexpr uint32_t kDimR = 1032;
  std::vector<int32_t> input(kDimA * kDimR, 1);
  input[0] = 2;
  input[1] = 2;
  auto output = RunReduce<int32_t, AscendC::Pattern::Reduce::AR>(
      input, kDimA, kDimA, kDimR, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<int32_t, AscendC::Pattern::Reduce::AR, false>(dst, src, tmp, shape, true);
      });
  EXPECT_EQ(output, (std::vector<int32_t>{4, 1}));
}

TEST(TestApiReduceExtend, ReduceProdFloatArNoReuseSourceCoversUnalignedTail) {
  constexpr uint32_t kDimA = 2;
  constexpr uint32_t kDimR = 72;
  std::vector<float> input(kDimA * kDimR, 1.0F);
  input.front() = 2.0F;
  input.back() = 3.0F;
  auto output = RunReduce<float, AscendC::Pattern::Reduce::AR>(
      input, kDimA, kDimA, kDimR, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<float, AscendC::Pattern::Reduce::AR, false>(dst, src, tmp, shape, true);
      });
  EXPECT_EQ(output, (std::vector<float>{2.0F, 3.0F}));
}

TEST(TestApiReduceExtend, ReduceProdInt64ArNoReuseSourceCoversAlignedAndUnalignedTails) {
  constexpr uint32_t kDimA = 2;
  auto makeInput = [](uint32_t dimR) {
    std::vector<int64_t> input(kDimA * dimR, 1);
    input[0] = 2;
    input[1] = 2;
    return input;
  };
  auto runReduce = [&makeInput](uint32_t dimR) {
    auto input = makeInput(dimR);
    return RunReduce<int64_t, AscendC::Pattern::Reduce::AR>(
        input, kDimA, kDimA, dimR, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
          AscendC::ReduceProdExtend<int64_t, AscendC::Pattern::Reduce::AR, false>(dst, src, tmp, shape, true);
        });
  };

  EXPECT_EQ(runReduce(1032), (std::vector<int64_t>{4, 1}));
  EXPECT_EQ(runReduce(1044), (std::vector<int64_t>{4, 1}));
}

TEST(TestApiReduceExtend, ReduceProdInt64CoversAlignedAndUnalignedB64Paths) {
  auto makeInput = [](uint32_t dimA, uint32_t dimR) {
    std::vector<int64_t> input(static_cast<size_t>(dimA) * dimR, 1);
    input.front() = 2;
    return input;
  };

  auto arReduce = [&makeInput](uint32_t dimA, uint32_t dimR) {
    auto input = makeInput(dimA, dimR);
    return RunReduce<int64_t, AscendC::Pattern::Reduce::AR>(
        input, dimA, dimA, dimR, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
          AscendC::ReduceProdExtend<int64_t, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
        });
  };
  auto raReduce = [&makeInput](uint32_t dimA, uint32_t dimR) {
    auto input = makeInput(dimA, dimR);
    return RunReduce<int64_t, AscendC::Pattern::Reduce::RA>(
        input, dimR, dimA, dimR, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
          AscendC::ReduceProdExtend<int64_t, AscendC::Pattern::Reduce::RA>(dst, src, tmp, shape, true);
        });
  };

  std::vector<int64_t> arExpected(2, 1);
  arExpected.front() = 2;
  EXPECT_EQ(arReduce(2, 32), arExpected);
  EXPECT_EQ(arReduce(1, 512), std::vector<int64_t>{2});
  {
    auto input = makeInput(2, 68);
    input.back() = 2;
    auto output = RunReduce<int64_t, AscendC::Pattern::Reduce::AR>(
        input, 2, 2, 68, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
          AscendC::ReduceProdExtend<int64_t, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
        });
    EXPECT_EQ(output, (std::vector<int64_t>{2, 2}));
  }
  {
    auto input = makeInput(2, 68);
    input[67] = 2;
    input[68] = 3;
    input.back() = 5;
    auto output = RunReduce<int64_t, AscendC::Pattern::Reduce::AR>(
        input, 2, 2, 68, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
          AscendC::ReduceProdExtend<int64_t, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
        });
    EXPECT_EQ(output, (std::vector<int64_t>{4, 15}));
  }

  std::vector<int64_t> raExpected(32, 1);
  raExpected.front() = 2;
  EXPECT_EQ(raReduce(8, 32), raExpected);
  raExpected.resize(64, 1);
  EXPECT_EQ(raReduce(8, 64), raExpected);
}

TEST(TestApiReduceExtend, ReduceProdInt64ArCoversLargeUnalignedReduceAxis) {
  constexpr uint32_t kDimA = 1;
  auto reduce = [](uint32_t dimR, int64_t lastValue) {
    std::vector<int64_t> input(dimR, 1);
    input.front() = 2;
    input.back() = lastValue;
    return RunReduce<int64_t, AscendC::Pattern::Reduce::AR>(
        input, kDimA, kDimA, dimR, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
          AscendC::ReduceProdExtend<int64_t, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
        });
  };

  EXPECT_EQ(reduce(4096, 2), (std::vector<int64_t>{4}));
  EXPECT_EQ(reduce(4164, 2), (std::vector<int64_t>{4}));
  EXPECT_EQ(reduce(4192, 2), (std::vector<int64_t>{4}));
  EXPECT_EQ(reduce(4164, -1), (std::vector<int64_t>{-2}));
  EXPECT_EQ(reduce(4224, -1), (std::vector<int64_t>{-2}));
}

TEST(TestApiReduceExtend, ReduceProdInt64ArCoversUnalignedVectorBoundaries) {
  constexpr uint32_t kDimA = 2;
  constexpr uint32_t kMaxChunkSize = 16384;
  auto reduce = [kMaxChunkSize](uint32_t dimR) {
    std::vector<int64_t> input(kDimA * dimR, 1);
    input.front() = 2;
    input[dimR - 1] = 3;
    input[dimR] = 5;
    input.back() = 7;
    std::vector<int64_t> output(kDimA, 1);
    for (uint32_t row = 0; row < kDimA; row++) {
      for (uint32_t offset = 0; offset < dimR; offset += kMaxChunkSize) {
        uint32_t chunkSize = std::min(kMaxChunkSize, dimR - offset);
        std::vector<int64_t> chunk(input.begin() + static_cast<int32_t>(row * dimR + offset),
                                   input.begin() + static_cast<int32_t>(row * dimR + offset + chunkSize));
        auto partial = RunReduce<int64_t, AscendC::Pattern::Reduce::AR>(
            chunk, 1, 1, chunkSize, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
              AscendC::ReduceProdExtend<int64_t, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
            });
        output[row] *= partial.front();
      }
    }
    return output;
  };

  for (uint32_t dimR : {36U, 64U, 68U, 129800U}) {
    EXPECT_EQ(reduce(dimR), (std::vector<int64_t>{6, 35})) << "dimR=" << dimR;
  }
}

TEST(TestApiReduceExtend, ReduceProdInt64ArPreservesVectorBoundaryElements) {
  auto reduceAt = [](uint32_t dimR, uint32_t index) {
    std::vector<int64_t> input(dimR, 1);
    input[index] = 2;
    return RunReduce<int64_t, AscendC::Pattern::Reduce::AR>(
        input, 1, 1, dimR, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
          AscendC::ReduceProdExtend<int64_t, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
        });
  };
  EXPECT_EQ(reduceAt(68, 31), (std::vector<int64_t>{2}));
  EXPECT_EQ(reduceAt(68, 32), (std::vector<int64_t>{2}));
  EXPECT_EQ(reduceAt(68, 63), (std::vector<int64_t>{2}));
  EXPECT_EQ(reduceAt(68, 64), (std::vector<int64_t>{2}));
  EXPECT_EQ(reduceAt(68, 67), (std::vector<int64_t>{2}));
  EXPECT_EQ(reduceAt(4164, 4095), (std::vector<int64_t>{2}));
  EXPECT_EQ(reduceAt(4164, 4096), (std::vector<int64_t>{2}));
  EXPECT_EQ(reduceAt(4164, 4163), (std::vector<int64_t>{2}));
  EXPECT_EQ(reduceAt(132, 64), (std::vector<int64_t>{2}));
  EXPECT_EQ(reduceAt(132, 127), (std::vector<int64_t>{2}));
  EXPECT_EQ(reduceAt(132, 128), (std::vector<int64_t>{2}));
}

TEST(TestApiReduceExtend, ReduceProdInt8ExecutionDtypeTruncatesWithoutSaturation) {
  std::vector<int16_t> input(64, 1);
  input[0] = 100;
  input[1] = 100;
  auto output = RunReduce<int16_t, AscendC::Pattern::Reduce::AR>(
      input, 2, 2, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<int16_t, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
      });
  EXPECT_EQ(output, (std::vector<int16_t>{10000, 1}));

  const std::vector<int16_t> source(4, 10000);
  const std::vector<uint8_t> uint8Expected(4, 16);
  const std::vector<uint8_t> uint8Actual = RunCast<int16_t, uint8_t>(source);
  EXPECT_EQ(uint8Actual, uint8Expected);
  const std::vector<int8_t> int8Expected(4, 16);
  const std::vector<int8_t> int8Actual = RunCast<uint8_t, int8_t>(uint8Expected);
  EXPECT_EQ(int8Actual, int8Expected);
}

TEST(TestApiReduceExtend, ReduceProdInt8ExecutionDtypeRaNonPowerOfTwo) {
  std::vector<int16_t> input(96, 1);
  input[0] = 100;
  input[1] = 100;
  input[32] = 2;
  input[33] = 2;
  auto output = RunReduce<int16_t, AscendC::Pattern::Reduce::RA>(
      input, 32, 3, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<int16_t, AscendC::Pattern::Reduce::RA>(dst, src, tmp, shape, true);
      });
  std::vector<int16_t> expected(32, 1);
  expected[0] = 200;
  expected[1] = 200;
  EXPECT_EQ(output, expected);
}

TEST(TestApiReduceExtend, ReduceProdInt16RaCoverE2eShapeAndOverflow) {
  std::vector<int16_t> input(192, 1);
  input[0] = 100;
  input[1] = 100;
  input[32] = 100;
  input[33] = 100;
  input[64] = 100;
  input[65] = 100;
  auto output = RunReduce<int16_t, AscendC::Pattern::Reduce::RA>(
      input, 32, 6, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<int16_t, AscendC::Pattern::Reduce::RA>(dst, src, tmp, shape, true);
      });
  std::vector<int16_t> expected(32, 1);
  expected[0] = static_cast<int16_t>(1000000);
  expected[1] = static_cast<int16_t>(1000000);
  EXPECT_EQ(output, expected);
}

TEST(TestApiReduceExtend, ReduceProdInt16RaLargeOutputCoverE2eShape) {
  constexpr uint32_t outputCount = 128U;
  constexpr uint32_t reduceCount = 3U;
  std::vector<int16_t> input(outputCount * reduceCount, 1);
  for (uint32_t offset = 0U; offset < 3U; offset++) {
    input[offset] = 100;
    input[outputCount + offset] = 100;
    input[2U * outputCount + offset] = 100;
  }
  auto output = RunReduce<int16_t, AscendC::Pattern::Reduce::RA>(
      input, outputCount, reduceCount, outputCount, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<int16_t, AscendC::Pattern::Reduce::RA>(dst, src, tmp, shape, true);
      });
  std::vector<int16_t> expected(outputCount, 1);
  constexpr int16_t overflowExpected = static_cast<int16_t>(1000000);
  expected[0] = overflowExpected;
  expected[1] = overflowExpected;
  expected[2] = overflowExpected;
  EXPECT_EQ(output, expected);
}

TEST(TestApiReduceExtend, ReduceProdInt32RaCoverE2eShape) {
  std::vector<int32_t> input(192, 1);
  input[0] = 100;
  input[1] = 100;
  input[32] = 100;
  input[33] = 100;
  input[64] = 100;
  input[65] = 100;
  auto output = RunReduce<int32_t, AscendC::Pattern::Reduce::RA>(
      input, 32, 6, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<int32_t, AscendC::Pattern::Reduce::RA>(dst, src, tmp, shape, true);
      });
  std::vector<int32_t> expected(32, 1);
  expected[0] = 1000000;
  expected[1] = 1000000;
  EXPECT_EQ(output, expected);
}

TEST(TestApiReduceExtend, ReduceProdInt32RaCoverOutputLargerThanVectorLength) {
  constexpr uint32_t outputCount = 64U;
  constexpr uint32_t reduceCount = 33U;
  std::vector<int32_t> input(outputCount * reduceCount, 1);
  input[0] = 2;
  input[outputCount + 1] = 3;
  input[2U * outputCount + 2] = 4;
  input[3U * outputCount + 3] = 5;
  auto output = RunReduce<int32_t, AscendC::Pattern::Reduce::RA>(
      input, outputCount, reduceCount, outputCount, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<int32_t, AscendC::Pattern::Reduce::RA>(dst, src, tmp, shape, true);
      });
  std::vector<int32_t> expected(outputCount, 1);
  expected[0] = 2;
  expected[1] = 3;
  expected[2] = 4;
  expected[3] = 5;
  EXPECT_EQ(output, expected);
}

TEST(TestApiReduceExtend, ReduceProdFloatRaNoReuseSourceKeepsTwoRows) {
  constexpr uint32_t outputCount = 128U;
  constexpr uint32_t reduceCount = 2U;
  std::vector<float> input(outputCount * reduceCount, 1.0F);
  input.front() = 2.0F;
  input[outputCount + 1U] = 3.0F;
  auto output = RunReduce<float, AscendC::Pattern::Reduce::RA>(
      input, outputCount, reduceCount, outputCount, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceProdExtend<float, AscendC::Pattern::Reduce::RA, false>(dst, src, tmp, shape, true);
      });
  std::vector<float> expected(outputCount, 1.0F);
  expected[0] = 2.0F;
  expected[1] = 3.0F;
  EXPECT_EQ(output, expected);
}

template <typename T, typename Pattern>
void TestProdNoReuseSourcePreservesUnalignedSource(uint32_t dim0, uint32_t dim1) {
  std::vector<T> input(dim0 * dim1, T{1});
  input.front() = T{2};
  const std::vector<T> expectedInput = input;
  const uint32_t inputBytes = AlignUp(input.size() * sizeof(T), ONE_BLK_SIZE);
  const uint32_t outputCount = std::is_same_v<Pattern, AscendC::Pattern::Reduce::AR> ? dim0 : dim1;
  const uint32_t outputBytes = AlignUp(outputCount * sizeof(T), ONE_BLK_SIZE);
  auto *inputGm = static_cast<T *>(AscendC::GmAlloc(inputBytes));
  auto *outputGm = static_cast<T *>(AscendC::GmAlloc(outputBytes));
  auto *sourceAfterGm = static_cast<T *>(AscendC::GmAlloc(inputBytes));
  std::copy(input.begin(), input.end(), inputGm);
  std::fill_n(outputGm, outputBytes / sizeof(T), T{});
  auto kernel = [](T *kernelInput, T *kernelOutput, T *kernelSourceAfter, uint32_t shape0, uint32_t shape1) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuffer;
    TBuf<TPosition::VECCALC> dstBuffer;
    TBuf<TPosition::VECCALC> tmpBuffer;
    pipe.InitBuffer(srcBuffer, AlignUp(shape0 * shape1 * sizeof(T), ONE_BLK_SIZE));
    pipe.InitBuffer(
        dstBuffer,
        AlignUp((std::is_same_v<Pattern, AscendC::Pattern::Reduce::AR> ? shape0 : shape1) * sizeof(T), ONE_BLK_SIZE));
    pipe.InitBuffer(tmpBuffer, AlignUp(shape0 * shape1 * sizeof(T), ONE_BLK_SIZE));

    LocalTensor<T> src = srcBuffer.Get<T>();
    LocalTensor<T> dst = dstBuffer.Get<T>();
    LocalTensor<uint8_t> tmp = tmpBuffer.Get<uint8_t>();
    GmToUb(src, kernelInput, shape0 * shape1);
    if constexpr (std::is_same_v<Pattern, AscendC::Pattern::Reduce::AR>) {
      const uint32_t shape[] = {shape0, shape1};
      AscendC::ReduceProdExtend<T, AscendC::Pattern::Reduce::AR, false>(dst, src, tmp, shape, true);
    } else {
      const uint32_t shape[] = {shape0, shape1};
      __ubuf__ T *srcAddr = (__ubuf__ T *)src.GetPhyAddr();
      __ubuf__ T *dstAddr = (__ubuf__ T *)dst.GetPhyAddr();
      __ubuf__ T *tmpAddr = (__ubuf__ T *)tmp.GetPhyAddr();
      if constexpr (sizeof(T) == 8) {
        AscendC::ReduceExtendInternal::Internal::ReduceProdRAB64ReuseSourceCompute<T, false>(dstAddr, srcAddr, tmpAddr,
                                                                                             shape);
      } else {
        AscendC::ReduceExtendInternal::Internal::ReduceProdRAReuseSourceCompute<T, false>(dstAddr, srcAddr, tmpAddr,
                                                                                          shape);
      }
    }
    UbToGm(kernelOutput, dst, std::is_same_v<Pattern, AscendC::Pattern::Reduce::AR> ? shape0 : shape1);
    UbToGm(kernelSourceAfter, src, shape0 * shape1);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, inputGm, outputGm, sourceAfterGm, dim0, dim1);
  const std::vector<T> output(outputGm, outputGm + outputCount);
  AscendC::GmFree(inputGm);
  AscendC::GmFree(outputGm);
  const std::vector<T> sourceAfter(sourceAfterGm, sourceAfterGm + input.size());
  AscendC::GmFree(sourceAfterGm);

  std::vector<T> expectedOutput(outputCount, T{1});
  expectedOutput.front() = T{2};
  EXPECT_EQ(output, expectedOutput);
  EXPECT_EQ(sourceAfter, expectedInput);
}

template <typename T>
std::vector<T> RunLogicalReduceWithSourceAfter(const std::vector<T> &input, uint32_t outputCount, uint32_t dim0,
                                               uint32_t dim1, bool isAny, bool isAr,
                                               std::vector<T> &sourceAfterOutput) {
  const uint32_t inputBytes = AlignUp(input.size() * sizeof(T), ONE_BLK_SIZE);
  const uint32_t outputBytes = AlignUp(outputCount * sizeof(T), ONE_BLK_SIZE);
  auto *inputGm = static_cast<T *>(AscendC::GmAlloc(inputBytes));
  auto *outputGm = static_cast<T *>(AscendC::GmAlloc(outputBytes));
  auto *sourceAfterGm = static_cast<T *>(AscendC::GmAlloc(inputBytes));
  std::copy(input.begin(), input.end(), inputGm);
  std::fill_n(outputGm, outputBytes / sizeof(T), T{});

  auto kernel = [isAny, isAr](T *kernelInput, T *kernelOutput, T *kernelSourceAfter, uint32_t shape0, uint32_t shape1) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuffer;
    TBuf<TPosition::VECCALC> dstBuffer;
    TBuf<TPosition::VECCALC> tmpBuffer;
    pipe.InitBuffer(srcBuffer, AlignUp(shape0 * shape1 * sizeof(T), ONE_BLK_SIZE));
    pipe.InitBuffer(dstBuffer, AlignUp((isAr ? shape0 : shape1) * sizeof(T), ONE_BLK_SIZE));
    pipe.InitBuffer(tmpBuffer, AlignUp(shape0 * shape1 * sizeof(T), ONE_BLK_SIZE));

    LocalTensor<T> src = srcBuffer.Get<T>();
    LocalTensor<T> dst = dstBuffer.Get<T>();
    LocalTensor<uint8_t> tmp = tmpBuffer.Get<uint8_t>();
    GmToUb(src, kernelInput, shape0 * shape1);
    const uint32_t shape[] = {shape0, shape1};
    if (isAny) {
      if (isAr) {
        AscendC::ReduceExtendInternal::Internal::ReduceAnyB64ARCompute<T, false>(
            (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), (__ubuf__ T *)tmp.GetPhyAddr(), shape);
      } else {
        AscendC::ReduceExtendInternal::Internal::ReduceAnyB64RACompute<T, false>(
            (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), (__ubuf__ T *)tmp.GetPhyAddr(), shape);
      }
    } else if (isAr) {
      AscendC::ReduceExtendInternal::Internal::ReduceAllB64ARCompute<T, false>(
          (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), (__ubuf__ T *)tmp.GetPhyAddr(), shape);
    } else {
      AscendC::ReduceExtendInternal::Internal::ReduceAllB64RACompute<T, false>(
          (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), (__ubuf__ T *)tmp.GetPhyAddr(), shape);
    }
    UbToGm(kernelOutput, dst, isAr ? shape0 : shape1);
    UbToGm(kernelSourceAfter, src, shape0 * shape1);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, inputGm, outputGm, sourceAfterGm, dim0, dim1);

  std::vector<T> output(outputGm, outputGm + outputCount);
  sourceAfterOutput.assign(sourceAfterGm, sourceAfterGm + input.size());
  AscendC::GmFree(inputGm);
  AscendC::GmFree(outputGm);
  AscendC::GmFree(sourceAfterGm);
  return output;
}

TEST(TestApiReduceExtend, ReduceProdNoReuseSourcePreservesUnalignedSource) {
  TestProdNoReuseSourcePreservesUnalignedSource<float, AscendC::Pattern::Reduce::AR>(2, 72);
  TestProdNoReuseSourcePreservesUnalignedSource<float, AscendC::Pattern::Reduce::RA>(3, 33);
  TestProdNoReuseSourcePreservesUnalignedSource<int16_t, AscendC::Pattern::Reduce::AR>(2, 144);
  TestProdNoReuseSourcePreservesUnalignedSource<int16_t, AscendC::Pattern::Reduce::RA>(3, 33);
  TestProdNoReuseSourcePreservesUnalignedSource<int32_t, AscendC::Pattern::Reduce::AR>(2, 72);
  TestProdNoReuseSourcePreservesUnalignedSource<int32_t, AscendC::Pattern::Reduce::RA>(3, 33);
  TestProdNoReuseSourcePreservesUnalignedSource<int64_t, AscendC::Pattern::Reduce::AR>(2, 36);
  TestProdNoReuseSourcePreservesUnalignedSource<int64_t, AscendC::Pattern::Reduce::RA>(3, 6);
}

TEST(TestApiReduceExtend, ReduceAnyAllInt64NoReuseSourcePreservesUnalignedSource) {
  const std::vector<int64_t> anyInput{0, 1, 0, 0, 0, 0};
  const std::vector<int64_t> allInput{1, 0, 1, 1, 1, 1};
  const std::vector<int64_t> anyExpected{1, 0};
  const std::vector<int64_t> allExpected{0, 1};
  std::vector<int64_t> sourceAfter;
  auto anyArOutput = RunLogicalReduceWithSourceAfter<int64_t>(anyInput, 2U, 2U, 3U, true, true, sourceAfter);
  EXPECT_EQ(anyArOutput, anyExpected);
  EXPECT_EQ(sourceAfter, anyInput);

  auto anyRaOutput = RunLogicalReduceWithSourceAfter<int64_t>(anyInput, 3U, 2U, 3U, true, false, sourceAfter);
  EXPECT_EQ(anyRaOutput, (std::vector<int64_t>{0, 1, 0}));
  EXPECT_EQ(sourceAfter, anyInput);

  auto allArOutput = RunLogicalReduceWithSourceAfter<int64_t>(allInput, 2U, 2U, 3U, false, true, sourceAfter);
  EXPECT_EQ(allArOutput, allExpected);
  EXPECT_EQ(sourceAfter, allInput);

  auto allRaOutput = RunLogicalReduceWithSourceAfter<int64_t>(allInput, 3U, 2U, 3U, false, false, sourceAfter);
  EXPECT_EQ(allRaOutput, (std::vector<int64_t>{1, 0, 1}));
  EXPECT_EQ(sourceAfter, allInput);
}

TEST(TestApiReduceExtend, ReduceProdCopyUnalignedPreservesData) {
  using T = int16_t;
  constexpr uint32_t inputCount = 130U;
  std::vector<T> input(inputCount, T{1});
  input.front() = T{2};
  const uint32_t inputBytes = AlignUp(inputCount * sizeof(T), ONE_BLK_SIZE);
  auto *inputGm = static_cast<T *>(AscendC::GmAlloc(inputBytes));
  auto *outputGm = static_cast<T *>(AscendC::GmAlloc(inputBytes));
  std::copy(input.begin(), input.end(), inputGm);
  std::fill_n(outputGm, inputBytes / sizeof(T), T{});
  auto kernel = [](T *kernelInput, T *kernelOutput) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuffer;
    TBuf<TPosition::VECCALC> tmpBuffer;
    pipe.InitBuffer(srcBuffer, AlignUp(inputCount * sizeof(T), ONE_BLK_SIZE));
    pipe.InitBuffer(tmpBuffer, AlignUp(inputCount * sizeof(T), ONE_BLK_SIZE));
    LocalTensor<T> src = srcBuffer.Get<T>();
    LocalTensor<T> tmp = tmpBuffer.Get<T>();
    GmToUb(src, kernelInput, inputCount);
    AscendC::ReduceExtendInternal::Internal::ReduceProdCopyUnaligned<T>((__ubuf__ T *)tmp.GetPhyAddr(),
                                                                        (__ubuf__ T *)src.GetPhyAddr(), inputCount);
    UbToGm(kernelOutput, tmp, inputCount);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, inputGm, outputGm);
  const std::vector<T> output(outputGm, outputGm + inputCount);
  AscendC::GmFree(inputGm);
  AscendC::GmFree(outputGm);
  EXPECT_EQ(output, input);
}

template <typename T>
void TestLogicalReduceIntegerDtype(T zero, T one, T two, bool isSignedType = false) {
  std::vector<T> anyInput(64, zero);
  anyInput[7] = one;
  anyInput[32] = two;
  auto anyOutput = RunReduce<T, AscendC::Pattern::Reduce::AR>(
      anyInput, 2, 2, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceAnyExtend<T, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
      });
  EXPECT_EQ(anyOutput, isSignedType ? (std::vector<T>{zero, zero}) : (std::vector<T>{one, two}));

  std::vector<T> allInput(64, one);
  allInput[7] = zero;
  allInput[32] = zero;
  auto allOutput = RunReduce<T, AscendC::Pattern::Reduce::AR>(
      allInput, 2, 2, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceAllExtend<T, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape, true);
      });
  EXPECT_EQ(allOutput, isSignedType ? (std::vector<T>{one, one}) : (std::vector<T>{zero, zero}));

  auto anyRaOutput = RunReduce<T, AscendC::Pattern::Reduce::RA>(
      anyInput, 32, 2, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceAnyExtend<T, AscendC::Pattern::Reduce::RA>(dst, src, tmp, shape, true);
      });
  std::vector<T> anyRaExpected(32, zero);
  anyRaExpected[0] = isSignedType ? zero : two;
  anyRaExpected[7] = isSignedType ? zero : one;
  EXPECT_EQ(anyRaOutput, anyRaExpected);

  auto allRaOutput = RunReduce<T, AscendC::Pattern::Reduce::RA>(
      allInput, 32, 2, 32, [](auto &dst, auto &src, auto &tmp, const uint32_t *shape) {
        AscendC::ReduceAllExtend<T, AscendC::Pattern::Reduce::RA>(dst, src, tmp, shape, true);
      });
  std::vector<T> allRaExpected(32, one);
  allRaExpected[0] = isSignedType ? one : zero;
  allRaExpected[7] = isSignedType ? one : zero;
  EXPECT_EQ(allRaOutput, allRaExpected);
}

TEST(TestApiReduceExtend, ReduceAnyAllSupportDirectIntegerDtypes) {
  TestLogicalReduceIntegerDtype<int8_t>(int8_t{0}, int8_t{1}, int8_t{2});
  TestLogicalReduceIntegerDtype<int16_t>(int16_t{0}, int16_t{1}, int16_t{2});
  TestLogicalReduceIntegerDtype<int32_t>(int32_t{0}, int32_t{1}, int32_t{2});
  TestLogicalReduceIntegerDtype<int64_t>(int64_t{0}, int64_t{1}, int64_t{2});
}

TEST(TestApiReduceExtend, ReduceAnyAllSupportNegativeIntegerDtypes) {
  TestLogicalReduceIntegerDtype<int8_t>(int8_t{-1}, int8_t{-2}, int8_t{-3}, true);
  TestLogicalReduceIntegerDtype<int16_t>(int16_t{-1}, int16_t{-2}, int16_t{-3}, true);
  TestLogicalReduceIntegerDtype<int32_t>(int32_t{-1}, int32_t{-2}, int32_t{-3}, true);
  TestLogicalReduceIntegerDtype<int64_t>(int64_t{-1}, int64_t{-2}, int64_t{-3}, true);
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
