/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 按 codegen 生成的实参直接调用 Broadcast 包装函数，在 CPU 模拟器上验证带行距布局的数值。
//
// 源行距和内段填充由上游对齐策略决定。现有 E2E 构图尚未生成这两种布局：2026-11-15 验证的
// 多种 Load → Sum → Broadcast 形状均生成 last_dim_stride = 1。因此，本文件直接验证包装函数，
// 与 afir-broadcast-source-pitch.mlir 和 afir-broadcast-inner-pitch.mlir 的代码生成检查互补。
//
// 历史反向验证：形态 B 的第十个实参改为 1 后，会连续读取源的前四个元素（100、101、102、103），
// 而非下标 0、8、16、24 处的有效元素（100、108、116、124），逐元素断言随之失败。

#include <cstdint>

#include <gtest/gtest.h>

#include "kernel_operator.h"
#include "tikicpulib.h"

using namespace AscendC;

// 包装的头要在 `using namespace AscendC` 之后、且 `utils.h` 在前：`utils.h` 里的
// `KernelUtils::BlkSize` 用的是不带限定的 `ONE_BLK_SIZE`，而 `broadcast.h` 用的是
// `KernelUtils`。生成的 kernel 里的次序也是这个（ODS 的 `ApiExtensionHeader` 把 `utils.h`
// 记成前导头）。
#include "utils.h"

#include "broadcast.h"

namespace {
constexpr uint32_t kTileFloats = 256U;
constexpr uint32_t kTmpFloats = 1024U;
constexpr float kSentinel = -999.0F;
constexpr uint32_t kRows = 4U;
// 形态 B：源和目的地的行距均为 8 个元素（f32 的 32B 对齐单位），每行有 2 个有效元素。
constexpr uint32_t kSourcePitch = 8U;
constexpr uint32_t kValidColumns = 2U;
// 形态 A：内段 extent 2、步长 8，所以两侧的 `k` 都是 16。
constexpr uint32_t kInnerPitchK = 16U;
constexpr uint32_t kInnerStride = 8U;

struct ProbeBuffers {
  float *source = nullptr;
  float *destination = nullptr;
};

ProbeBuffers AllocateProbeBuffers(float base) {
  ProbeBuffers buffers;
  buffers.source = static_cast<float *>(AscendC::GmAlloc(kTileFloats * sizeof(float)));
  buffers.destination = static_cast<float *>(AscendC::GmAlloc(kTileFloats * sizeof(float)));
  if (buffers.source == nullptr || buffers.destination == nullptr) {
    return buffers;
  }
  for (uint32_t index = 0; index < kTileFloats; ++index) {
    buffers.source[index] = static_cast<float>(index) + base;
    // 预填输入中不存在的哨兵值，用于检测输出未写入的情况。
    buffers.destination[index] = kSentinel;
  }
  return buffers;
}

void FreeProbeBuffers(const ProbeBuffers &buffers) {
  if (buffers.source != nullptr) {
    AscendC::GmFree(buffers.source);
  }
  if (buffers.destination != nullptr) {
    AscendC::GmFree(buffers.destination);
  }
}
}  // namespace

// 将一块 UB 缓冲分为源张量、目的张量和包装函数所需的临时缓冲三部分。
#define AF_BROADCAST_PITCH_PROBE_BODY(call)                                            \
  TPipe pipe;                                                                          \
  TBuf<TPosition::VECCALC> buffer;                                                     \
  pipe.InitBuffer(buffer, (kTileFloats * 2U + kTmpFloats) * sizeof(float));            \
  LocalTensor<float> whole = buffer.Get<float>();                                      \
  LocalTensor<float> source = whole;                                                   \
  LocalTensor<float> destination = whole[kTileFloats];                                 \
  LocalTensor<uint8_t> tmp = whole[kTileFloats * 2U].ReinterpretCast<uint8_t>();       \
  GlobalTensor<float> sourceGm;                                                        \
  sourceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sourceAddress));           \
  GlobalTensor<float> destinationGm;                                                   \
  destinationGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(destinationAddress)); \
  DataCopy(source, sourceGm, kTileFloats);                                             \
  DataCopy(destination, destinationGm, kTileFloats);                                   \
  PipeBarrier<PIPE_ALL>();                                                             \
  call;                                                                                \
  PipeBarrier<PIPE_ALL>();                                                             \
  DataCopy(destinationGm, destination, kTileFloats)

// 形态 B [normal 4][brc 2]，源行距为 8，与 afir-broadcast-source-pitch.mlir 的预期调用一致。
extern "C" __global__ __aicore__ void AfBroadcastSourcePitchProbe(GM_ADDR sourceAddress, GM_ADDR destinationAddress) {
  AF_BROADCAST_PITCH_PROBE_BODY(
      Broadcast(destination, source, kRows, 1U, 0U, kRows, kSourcePitch, 0U, tmp, kSourcePitch));
}

// 形态 A [brc 4][normal 2]，两侧内段步长为 8，与 afir-broadcast-inner-pitch.mlir 的预期调用一致。
extern "C" __global__ __aicore__ void AfBroadcastInnerPitchProbe(GM_ADDR sourceAddress, GM_ADDR destinationAddress) {
  AF_BROADCAST_PITCH_PROBE_BODY(Broadcast(destination, source, 1U, kInnerPitchK, 0U, kRows, kInnerPitchK, 0U, tmp, 1U));
}

TEST(MlirE2eCpuBroadcastWrapper, SourcePitchReadsEveryRowStart) {
  const ProbeBuffers buffers = AllocateProbeBuffers(100.0F);
  if (buffers.source == nullptr || buffers.destination == nullptr) {
    FreeProbeBuffers(buffers);
    FAIL() << "probe buffer allocation failed";
  }
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AfBroadcastSourcePitchProbe, 1, reinterpret_cast<uint8_t *>(buffers.source),
              reinterpret_cast<uint8_t *>(buffers.destination));
  for (uint32_t row = 0; row < kRows; ++row) {
    // 源的有效元素按行距排布，所以第 row 行要读的是 `source[row * 8]`，不是 `source[row]`。
    const float expected = buffers.source[row * kSourcePitch];
    for (uint32_t column = 0; column < kValidColumns; ++column) {
      EXPECT_FLOAT_EQ(buffers.destination[row * kSourcePitch + column], expected)
          << "row=" << row << " column=" << column;
    }
  }
  FreeProbeBuffers(buffers);
}

TEST(MlirE2eCpuBroadcastWrapper, InnerPitchKeepsPaddedElementPositions) {
  const ProbeBuffers buffers = AllocateProbeBuffers(200.0F);
  if (buffers.source == nullptr || buffers.destination == nullptr) {
    FreeProbeBuffers(buffers);
    FAIL() << "probe buffer allocation failed";
  }
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(AfBroadcastInnerPitchProbe, 1, reinterpret_cast<uint8_t *>(buffers.source),
              reinterpret_cast<uint8_t *>(buffers.destination));
  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t element = 0; element < kValidColumns; ++element) {
      // 每行下标 0 和 8 为有效元素；同时复制的填充位置不参与比对。
      const float expected = buffers.source[element * kInnerStride];
      EXPECT_FLOAT_EQ(buffers.destination[row * kInnerPitchK + element * kInnerStride], expected)
          << "row=" << row << " element=" << element;
    }
  }
  FreeProbeBuffers(buffers);
}
