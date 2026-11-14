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
#include <cmath>
#include <memory>
#include <type_traits>
#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "tikicpulib.h"
// Match the extension-header prelude supplied by production kernel emission.
using namespace AscendC;
#include "ascendc/api/utils.h"
#include "ascendc/api/abs.h"
#include "ascendc/api/clipbyvalue.h"
#include "ascendc/api/cast.h"
#include "local_view.cpp"

namespace {
constexpr unsigned kCapacity = 64;
constexpr unsigned kRows = 4;
constexpr unsigned kColumns = 8;
constexpr unsigned kPitch = 16;
constexpr float kSentinel = -999.0F;

extern "C" __global__ __aicore__ void LocalViewKernel(GM_ADDR input, GM_ADDR output, unsigned operation) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffers[5];
  for (unsigned i = 0; i < 5; ++i) {
    pipe.InitBuffer(buffers[i], kCapacity * sizeof(float));
  }
  auto a = buffers[0].Get<float>();
  auto b = buffers[1].Get<float>();
  auto c = buffers[2].Get<float>();
  auto dst = buffers[3].Get<float>();
  auto tmp = buffers[4].Get<uint8_t>();
  AscendC::GlobalTensor<float> in, out;
  in.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
  out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
  AscendC::DataCopy(a, in, kCapacity);
  AscendC::DataCopy(b, in[kCapacity], kCapacity);
  AscendC::DataCopy(c, in[2 * kCapacity], kCapacity);
  AscendC::DataCopy(dst, out, kCapacity);
  AscendC::PipeBarrier<PIPE_ALL>();
  if (operation == 0) {
    abs_different_ub(a, dst, tmp);
  } else if (operation == 1) {
    add_different_ub(a, b, dst);
  } else {
    clip_different_ub(a, b, c, dst, tmp);
  }
  AscendC::PipeBarrier<PIPE_ALL>();
  AscendC::DataCopy(out, dst, kCapacity);
  AscendC::PipeBarrier<PIPE_ALL>();
}

TEST(MlirE2eCpuLocalView, IndependentStridesPreserveValuesAndPadding) {
  auto release = [](float *ptr) {
    if (ptr) {
      AscendC::GmFree(ptr);
    }
  };
  std::unique_ptr<float, decltype(release)> input(static_cast<float *>(AscendC::GmAlloc(3 * kCapacity * sizeof(float))),
                                                  release);
  std::unique_ptr<float, decltype(release)> output(static_cast<float *>(AscendC::GmAlloc(kCapacity * sizeof(float))),
                                                   release);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);
  std::fill_n(input.get(), 3 * kCapacity, kSentinel);
  for (unsigned row = 0; row < kRows; ++row) {
    for (unsigned col = 0; col < kColumns; ++col) {
      const unsigned logical = row * kColumns + col;
      input.get()[logical] = static_cast<float>(logical) - 16.0F;
      input.get()[kCapacity + row * kPitch + col] = -3.0F + row;
      input.get()[2 * kCapacity + logical] = 5.0F + row;
    }
  }
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  for (unsigned operation = 0; operation < 3; ++operation) {
    std::fill_n(output.get(), kCapacity, kSentinel);
    ICPU_RUN_KF(LocalViewKernel, 1, reinterpret_cast<uint8_t *>(input.get()), reinterpret_cast<uint8_t *>(output.get()),
                operation);
    for (unsigned index = 0; index < kCapacity; ++index) {
      const unsigned row = index / kPitch;
      const unsigned col = index % kPitch;
      float expected = kSentinel;
      if (col < kColumns) {
        const float a = input.get()[row * kColumns + col];
        const float b = input.get()[kCapacity + index];
        const float c = input.get()[2 * kCapacity + row * kColumns + col];
        expected = operation == 0 ? std::abs(a) : operation == 1 ? a + b : std::min(std::max(a, b), c);
      }
      EXPECT_FLOAT_EQ(output.get()[index], expected) << "operation=" << operation << " index=" << index;
    }
  }
}

template <typename Input>
__global__ __aicore__ void PaddedCastKernel(GM_ADDR input, GM_ADDR output) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> srcBuffer, dstBuffer, tmpBuffer;
  pipe.InitBuffer(srcBuffer, 128);
  pipe.InitBuffer(dstBuffer, 128);
  pipe.InitBuffer(tmpBuffer, 128);
  auto src = srcBuffer.Get<Input>();
  auto dst = dstBuffer.Get<float>();
  auto tmp = tmpBuffer.Get<uint8_t>();
  AscendC::GlobalTensor<Input> in;
  AscendC::GlobalTensor<float> out;
  in.SetGlobalBuffer(reinterpret_cast<__gm__ Input *>(input));
  out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
  AscendC::DataCopy(src, in, 64);
  AscendC::DataCopy(dst, out, 32);
  AscendC::PipeBarrier<PIPE_ALL>();
  if constexpr (std::is_same_v<Input, half>) {
    cast_padded_f16(src, dst, tmp);
  } else {
    cast_padded_bf16(src, dst, tmp);
  }
  AscendC::PipeBarrier<PIPE_ALL>();
  AscendC::DataCopy(out, dst, 32);
}

template <typename Input>
void CheckPaddedCast() {
  auto release = [](void *ptr) { AscendC::GmFree(ptr); };
  std::unique_ptr<Input, decltype(release)> input(static_cast<Input *>(AscendC::GmAlloc(128)), release);
  std::unique_ptr<float, decltype(release)> output(static_cast<float *>(AscendC::GmAlloc(128)), release);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);
  std::fill_n(input.get(), 64, static_cast<Input>(-99.0F));
  std::fill_n(output.get(), 32, kSentinel);
  for (unsigned row = 0; row < 4; ++row) {
    input.get()[row * 16] = static_cast<Input>(static_cast<float>(row) - 1.5F);
  }
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(PaddedCastKernel<Input>, 1, reinterpret_cast<uint8_t *>(input.get()),
              reinterpret_cast<uint8_t *>(output.get()));
  for (unsigned index = 0; index < 32; ++index) {
    const float expected = index % 8 == 0 ? static_cast<float>(index / 8) - 1.5F : kSentinel;
    EXPECT_FLOAT_EQ(output.get()[index], expected) << "index=" << index;
  }
}

TEST(MlirE2eCpuLocalView, PaddedCastPreservesValuesAndPadding) {
  CheckPaddedCast<half>();
  CheckPaddedCast<bfloat16_t>();
}
}  // namespace
