/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- VfEmissionFormProbe.cpp - VF 的两种生成形态哪种收得下 ----------------===//
//
// 这不是测试，是一个测量工具（与同目录的 `CapabilityBoundaryProbe.cpp` /
// `legacy_takeover_table.sh` 同一性质）。它不接任何 `check-*` 目标，要量就手动跑：
//
//     bash autofuse/mlir/test/probe/vf_emission_form/run.sh
//
// 要回答的问题来自 `docs/development/2026-10-18-afir-vector-func-implementation-design.md`
// 的「假设和依赖关系」第 1 条：VF 那一段代码有两种输出形态，选哪一种决定了实现的代价。
//
//   形态一 B2  legacy 当前的做法：外联一个 `inline __simd_vf__` 函数，kernel 生成调用。
//              要多一个外联 pass、一条 dims/strides 参数表、一个架构 `#if` 的生成点
//              （而架构门的字面量不许写在 artifact 层，见 check_platform_agnostic_layers.sh）。
//   形态二 B1  PyAsc 上游的做法（`emitasc.vec_scope` / `vf_for` 已经建模且已能生成）：
//              就地 `__VEC_SCOPE__ { for (...) { ... } }`，直接用 kernel 自己的 SSA 值，
//              上面三件一件都不要。
//
// 两种形态算同一件事（`y = |a + b|`，256 个 f32），所以**数值一比就知道这一档 CANN +
// 这一代 arch 收不收**。输出缓冲预置哨兵 `-12345.0f` 而不是 0：不然「整段没被编进去」
// （架构门没命中时 legacy 的产物就是这样，函数与调用一起被预处理器吃掉）会看起来像
// 「算出来是 0」。
//
// 两档都要跑，因为它们答的是两个问题：
//
//   `run.sh`           CPU 模拟器（`g++` + tikicpulib）跑数值 —— 形态**算得对不对**；
//   `run.sh --device`  真实 NPU 编译器（`ccec --cce-aicore-arch=dav-c310`）只编不跑
//                      —— 形态编得过吗、编出来的是不是向量指令。
//
// 第二档是必须的：模拟器那一份是 `g++` 编的，它收得下不等于 ccec 收得下。设备侧这一档
// 用 `-DVF_PROBE_DEVICE_ONLY` 把 host 侧的 `main` 与 tikicpulib 关掉（ccec 编不了它们）。
//
// 2026-10-18 实测（CANN 9.2.0）：
//   - 模拟器：两种形态都 `mismatches=0`；
//   - ccec（dav-c310）：两种形态都编得过，而且反汇编出来是同一段向量代码——
//     `__VEC_SCOPE__` 被编译器外联成一个 `.vector.thread` 函数，指令直方图与 `__simd_vf__`
//     那一份逐项相同（`VAG_V310` 1 / `VLOOPV2_V310` 1 / `VLD` 2 / `PLT.b32` 1 /
//     `VADD.f32` 1 / `VABS.f32` 1 / `VST` 1 / `SEND` 1），标量侧各生成一条 `VF` 去启动它。
//     也就是说两种形态的差别只在源码层，不在指令层。
//
// 这个测量工具仍然答不了的：上板实测（功耗、真实时序），以及形态二在更复杂的子图
// （多输出、跨 dtype、谓词进 Select）上是否还等价——那要靠 E2E 一格一格验。
//
//===----------------------------------------------------------------------===//

#if !defined(VF_PROBE_DEVICE_ONLY)
#include <cstdio>

#include "tikicpulib.h"
#endif
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t kCount = 256;

#if defined(__DAV_C310__) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
// 形态一：legacy 的外联 __simd_vf__ 函数。
inline __simd_vf__ void VfAddAbs(__local_mem__ float *dst, __local_mem__ float *a, __local_mem__ float *b,
                                 uint32_t count) {
  constexpr static uint32_t VECTOR_LENGTH = AscendC::GetVecLen();
  constexpr static uint32_t ELEMENT_PER_VECTOR_LENGTH = VECTOR_LENGTH / sizeof(float);
  uint16_t loop_times = static_cast<uint16_t>((count + ELEMENT_PER_VECTOR_LENGTH - 1) / ELEMENT_PER_VECTOR_LENGTH);
  AscendC::MicroAPI::RegTensor<float> v0;
  AscendC::MicroAPI::RegTensor<float> v1;
  AscendC::MicroAPI::RegTensor<float> v2;
  uint32_t sreg = count;
  AscendC::MicroAPI::MaskReg preg;
  for (uint16_t i = 0; i < loop_times; i++) {
    preg = AscendC::MicroAPI::UpdateMask<float>(sreg);
    AscendC::MicroAPI::LoadAlign(v0, a + i * ELEMENT_PER_VECTOR_LENGTH);
    AscendC::MicroAPI::LoadAlign(v1, b + i * ELEMENT_PER_VECTOR_LENGTH);
    AscendC::MicroAPI::Add(v2, v0, v1, preg);
    AscendC::MicroAPI::Abs(v2, v2, preg);
    AscendC::MicroAPI::StoreAlign(dst + i * ELEMENT_PER_VECTOR_LENGTH, v2, preg);
  }
}
#endif

extern "C" __global__ __aicore__ void KernelSimdVf(GM_ADDR ga, GM_ADDR gb, GM_ADDR gy) {
  TPipe tpipe;
  TBuf<TPosition::VECCALC> abuf, bbuf, ybuf;
  tpipe.InitBuffer(abuf, sizeof(float) * kCount);
  tpipe.InitBuffer(bbuf, sizeof(float) * kCount);
  tpipe.InitBuffer(ybuf, sizeof(float) * kCount);
  LocalTensor<float> la = abuf.Get<float>();
  LocalTensor<float> lb = bbuf.Get<float>();
  LocalTensor<float> ly = ybuf.Get<float>();
  GlobalTensor<float> a;
  a.SetGlobalBuffer((__gm__ float *)ga);
  GlobalTensor<float> b;
  b.SetGlobalBuffer((__gm__ float *)gb);
  GlobalTensor<float> y;
  y.SetGlobalBuffer((__gm__ float *)gy);
  DataCopy(la, a, kCount);
  DataCopy(lb, b, kCount);
#if defined(__DAV_C310__) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
  VfAddAbs((__local_mem__ float *)ly[0].GetPhyAddr(), (__local_mem__ float *)la[0].GetPhyAddr(),
           (__local_mem__ float *)lb[0].GetPhyAddr(), kCount);
#endif
  DataCopy(y, ly, kCount);
}

extern "C" __global__ __aicore__ void KernelVecScope(GM_ADDR ga, GM_ADDR gb, GM_ADDR gy) {
  TPipe tpipe;
  TBuf<TPosition::VECCALC> abuf, bbuf, ybuf;
  tpipe.InitBuffer(abuf, sizeof(float) * kCount);
  tpipe.InitBuffer(bbuf, sizeof(float) * kCount);
  tpipe.InitBuffer(ybuf, sizeof(float) * kCount);
  LocalTensor<float> la = abuf.Get<float>();
  LocalTensor<float> lb = bbuf.Get<float>();
  LocalTensor<float> ly = ybuf.Get<float>();
  GlobalTensor<float> a;
  a.SetGlobalBuffer((__gm__ float *)ga);
  GlobalTensor<float> b;
  b.SetGlobalBuffer((__gm__ float *)gb);
  GlobalTensor<float> y;
  y.SetGlobalBuffer((__gm__ float *)gy);
  DataCopy(la, a, kCount);
  DataCopy(lb, b, kCount);
#if defined(VEC_SCOPE_VARIANT)
  {
    constexpr static uint32_t ELEMENT_PER_VECTOR_LENGTH = AscendC::GetVecLen() / sizeof(float);
    uint16_t loop_times = static_cast<uint16_t>((kCount + ELEMENT_PER_VECTOR_LENGTH - 1) / ELEMENT_PER_VECTOR_LENGTH);
    __local_mem__ float *dst = (__local_mem__ float *)ly[0].GetPhyAddr();
    __local_mem__ float *pa = (__local_mem__ float *)la[0].GetPhyAddr();
    __local_mem__ float *pb = (__local_mem__ float *)lb[0].GetPhyAddr();
    uint32_t sreg = kCount;
    __VEC_SCOPE__ {
      AscendC::MicroAPI::RegTensor<float> v0;
      AscendC::MicroAPI::RegTensor<float> v1;
      AscendC::MicroAPI::RegTensor<float> v2;
      AscendC::MicroAPI::MaskReg preg;
      for (uint16_t i = 0; i < loop_times; i++) {
        preg = AscendC::MicroAPI::UpdateMask<float>(sreg);
        AscendC::MicroAPI::LoadAlign(v0, pa + i * ELEMENT_PER_VECTOR_LENGTH);
        AscendC::MicroAPI::LoadAlign(v1, pb + i * ELEMENT_PER_VECTOR_LENGTH);
        AscendC::MicroAPI::Add(v2, v0, v1, preg);
        AscendC::MicroAPI::Abs(v2, v2, preg);
        AscendC::MicroAPI::StoreAlign(dst + i * ELEMENT_PER_VECTOR_LENGTH, v2, preg);
      }
    }
  }
#endif
  DataCopy(y, ly, kCount);
}

#if !defined(VF_PROBE_DEVICE_ONLY)
int main() {
  float *a = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kCount));
  float *b = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kCount));
  float *y = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kCount));
  for (uint32_t i = 0; i < kCount; ++i) {
    a[i] = static_cast<float>(i) - 100.0f;
    b[i] = 0.5f;
    y[i] = -12345.0f;  // 哨兵：不被写就能看出来
  }
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
#if defined(VEC_SCOPE_VARIANT)
  ICPU_RUN_KF(KernelVecScope, 1, (GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)y);
#else
  ICPU_RUN_KF(KernelSimdVf, 1, (GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)y);
#endif
  uint32_t bad = 0;
  for (uint32_t i = 0; i < kCount; ++i) {
    const float expect = (a[i] + b[i]) < 0.0f ? -(a[i] + b[i]) : (a[i] + b[i]);
    if (y[i] != expect) {
      if (bad < 5) {
        printf("diff[%u]: got %f want %f\n", i, y[i], expect);
      }
      ++bad;
    }
  }
  printf("%s: mismatches=%u / %u\n", (bad == 0 ? "PASS" : "FAIL"), bad, kCount);
  AscendC::GmFree(a);
  AscendC::GmFree(b);
  AscendC::GmFree(y);
  return bad == 0 ? 0 : 1;
}
#endif  // !VF_PROBE_DEVICE_ONLY
