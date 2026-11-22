/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/gather.h"
#include <limits>
#include <type_traits>

using namespace AscendC;

namespace {
constexpr uint32_t kBufferSize = 32;
constexpr uint32_t kDstOffset = 1;
constexpr uint64_t kCopyLength = 1;
constexpr uint8_t kSentinel = 0xA5;
constexpr uint8_t kSourceValue = 0x3C;
}  // namespace

TEST(TestApiGather, DataCopySimdSimtSmallLengthWithinPadding) {
  auto *src = static_cast<uint8_t *>(AscendC::GmAlloc(kBufferSize));
  auto *dst = static_cast<uint8_t *>(AscendC::GmAlloc(kBufferSize));
  for (uint32_t i = 0; i < kBufferSize; ++i) {
    src[i] = kSourceValue + i;
    dst[i] = kSentinel;
  }

  auto kernel = [](uint8_t *src, uint8_t *dst) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> dst_buf;
    pipe.InitBuffer(dst_buf, kBufferSize);
    LocalTensor<uint8_t> local_dst = dst_buf.Get<uint8_t>();
    GmToUb(local_dst, dst, kBufferSize);

    GlobalTensor<uint8_t> global_src;
    global_src.SetGlobalBuffer((__gm__ uint8_t *)src);
    DataCopySimdSimt(local_dst, global_src, kDstOffset, 0, kCopyLength, false);
    UbToGm(dst, local_dst, kBufferSize);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, src, dst);

  EXPECT_EQ(dst[kDstOffset], kSourceValue);
  for (uint32_t i = 0; i < kBufferSize; ++i) {
    if (i != kDstOffset) {
      EXPECT_EQ(dst[i], kSentinel) << "unexpected write at index " << i;
    }
  }

  AscendC::GmFree(src);
  AscendC::GmFree(dst);
}

TEST(TestApiGather, Int64VRegRejectsInsufficientWorkspaceBeforeAccessingTensors) {
  LocalTensor<int64_t> dst;
  LocalTensor<uint8_t> tmp;
  GlobalTensor<int64_t> params;
  GlobalTensor<int64_t> indices;
  // These are the former heuristic's zero-remainder and underflow cases.
  EXPECT_FALSE((GatherExtendVReg<int64_t, int64_t, false>(0, tmp, 28160, dst, params, indices, 65, 225280)));
  EXPECT_FALSE((GatherExtendVReg<int64_t, int64_t, false>(0, tmp, 30000, dst, params, indices, 65, 237056)));
  // Params fit, but a full register of indices and the conversion buffer do not.
  EXPECT_FALSE((GatherExtendVReg<int64_t, int64_t, false>(0, tmp, 1025, dst, params, indices, 65, 8288)));
}

namespace {
template <typename ValueT, typename IndexT, uint32_t output_size = 257>
void CheckGatherWithWorkspace(uint32_t param_size, uint32_t tmp_size) {
  constexpr uint32_t offset = 13;
  constexpr uint32_t guard_size = 64;
  auto *params = static_cast<ValueT *>(AscendC::GmAlloc(param_size * sizeof(ValueT)));
  auto *indices = static_cast<IndexT *>(AscendC::GmAlloc((output_size + offset) * sizeof(IndexT)));
  auto *output = static_cast<ValueT *>(AscendC::GmAlloc(output_size * sizeof(ValueT)));
  auto *guard = static_cast<uint8_t *>(AscendC::GmAlloc(guard_size + 1));
  const int64_t values[] = {0,
                            -1,
                            (int64_t{1} << 53) + 1,
                            -(int64_t{1} << 53) - 1,
                            std::numeric_limits<int64_t>::min(),
                            std::numeric_limits<int64_t>::max()};
  for (uint32_t i = 0; i < param_size; ++i) {
    if constexpr (std::is_same<ValueT, int64_t>::value) {
      params[i] = values[i % (sizeof(values) / sizeof(values[0]))];
    } else {
      params[i] = static_cast<ValueT>(static_cast<int32_t>(i % 97) - 48);
    }
  }
  for (uint32_t i = 0; i < output_size + offset; ++i) {
    indices[i] = static_cast<IndexT>((i * 37) % param_size);
  }
  auto kernel = [](ValueT *params, IndexT *indices, ValueT *output, uint8_t *guard, uint32_t param_size,
                   uint32_t tmp_size) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> dst_buf, tmp_buf;
    pipe.InitBuffer(dst_buf, (output_size * sizeof(ValueT) + 31) / 32 * 32);
    pipe.InitBuffer(tmp_buf, tmp_size + guard_size);
    auto dst = dst_buf.Get<ValueT>();
    auto tmp = tmp_buf.Get<uint8_t>();
    for (uint32_t i = 0; i < guard_size; ++i) {
      tmp.SetValue(tmp_size + i, kSentinel);
    }
    GlobalTensor<ValueT> src;
    GlobalTensor<IndexT> idx;
    src.SetGlobalBuffer(params);
    idx.SetGlobalBuffer(indices);
    if constexpr (std::is_same<ValueT, int64_t>::value) {
      GatherExtend<ValueT, IndexT, CASE1, 1, false>(dst, src, idx, output_size, offset, output_size + offset,
                                                    param_size, 1, tmp, tmp_size, param_size, 1, output_size, 1, 1);
      guard[guard_size] = 1;
    } else {
      // Exercise the changed VReg layout directly for the existing value types.
      guard[guard_size] = GatherExtendVReg<ValueT, IndexT, false>(offset, tmp, param_size, dst, src, idx[offset],
                                                                  output_size, tmp_size);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    UbToGm(output, dst, output_size);
    for (uint32_t i = 0; i < guard_size; ++i) {
      guard[i] = tmp.GetValue(tmp_size + i);
    }
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, params, indices, output, guard, param_size, tmp_size);
  EXPECT_EQ(guard[guard_size], 1);
  for (uint32_t i = 0; i < output_size; ++i) {
    EXPECT_EQ(output[i], params[indices[i + offset]]) << "output index " << i;
  }
  for (uint32_t i = 0; i < guard_size; ++i) {
    EXPECT_EQ(guard[i], kSentinel) << "scratch overwrite at byte " << i;
  }
  AscendC::GmFree(params);
  AscendC::GmFree(indices);
  AscendC::GmFree(output);
  AscendC::GmFree(guard);
}
}  // namespace

TEST(TestApiGather, Int64VRegHandlesAlignedWorkspaceChunksAndTail) {
  // Unaligned params payload, multiple index chunks, and a one-element tail.
  CheckGatherWithWorkspace<int64_t, int64_t>(1025, 9216);
  CheckGatherWithWorkspace<int64_t, int32_t>(1025, 9216);
}

TEST(TestApiGather, Int64GatherFallsBackWhenFullLoadWorkspaceDoesNotFit) {
  CheckGatherWithWorkspace<int64_t, int64_t>(28160, 32);
  CheckGatherWithWorkspace<int64_t, int64_t>(30000, 32);
  // Above the VReg selection threshold, but still too small for params.
  CheckGatherWithWorkspace<int64_t, int64_t>(30000, 8224);
}

TEST(TestApiGather, ExistingValueTypesKeepScratchWithinAlignedCapacity) {
  // The old int32-index calculation used all remaining bytes without subtracting
  // its extra 32-byte gap after the params payload, overwriting the guard.
  CheckGatherWithWorkspace<float, int32_t, 641>(4096, 17664);
  CheckGatherWithWorkspace<half, int32_t, 641>(8192, 17664);
  CheckGatherWithWorkspace<float, int64_t, 641>(4096, 18432);
  CheckGatherWithWorkspace<half, int64_t, 641>(8192, 18432);
}
