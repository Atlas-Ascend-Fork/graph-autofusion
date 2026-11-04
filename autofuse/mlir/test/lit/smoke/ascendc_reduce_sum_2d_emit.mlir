// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `ascendc.reduce_sum_2d_l2` 发射到 C++ 文本这一层此前一条用例都没有，而它唯一会出错的
// 地方恰恰只在这一层：
//
// `srcShape` 在 ODS 上是 `Variadic<I32>`，而 MLIR 的 `i32` 是 signless，
// `CodeEmitter::shouldMapToUnsigned` 把 signless 与 signed 一并映射成**有符号**，所以这些
// 值落到 C++ 里是 `int32_t`。数组却必须是 `uint32_t[]`——API 签名是
// `const uint32_t srcShape[]`（`reduce.h:233`），改不了。往 `uint32_t[]` 的花括号初始化
// 列表里塞 `int32_t` 是 C++11 起禁止的窄化转换（[dcl.init.list]/7：源类型的负值目标类型
// 表示不了，且这些是运行期值、不是值域内的常量表达式），clang 系的 bisheng 默认把
// `-Wc++11-narrowing` 报成 **error**。
//
// 为什么非得钉在发射文本上：这条只有真机编 kernel 时才会撞上。主机侧编译不碰 device 代码，
// CPU E2E 也不走这个重载，于是容器里 ctest / lit 全绿、仓库看着健康，别人一上设备就挂在
// 一个他不知道要打补丁的地方。「跑得通」在这里不是判据。
//
// 为什么是这个形态而不是接在 `conversion/afir-to-ascendc-reduce-sum.mlir` 后面：那份输入是
// 真实的 plan 模块，跑完流水线的产物里有 `afplan.*` 属性和 `index.ceildivu`，而
// `ascir-translate` 两个方言都不注册，翻译不了。发射器的判据不需要整条流水线，直接给它
// `ascendc.*` 就够。

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// AR：归约 `[A, R]` 的末维。
func.func @reduce_sum_2d_ar(%dst: !ascendc.local_tensor<4x1xf32>, %src: !ascendc.local_tensor<4x8xf32>,
                            %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %r: i32) {
  ascendc.reduce_sum_2d_l2 %dst, %src, %tmp, %a, %r {layout = 0 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xf32>, !ascendc.local_tensor<4x8xf32>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}

// 两项都要带 `static_cast<uint32_t>`，少一项就不是合法的 C++11。
// CHECK-LABEL: reduce_sum_2d_ar
// CHECK: uint32_t {{[a-zA-Z0-9_]+}}_reduce_shape[] = {static_cast<uint32_t>({{[a-zA-Z0-9_]+}}), static_cast<uint32_t>({{[a-zA-Z0-9_]+}})};
// CHECK: ReduceSum<float, AscendC::Pattern::Reduce::AR, false>

// -----

// RA：归约 `[R, A]` 的首维。走的是同一个 printer，这里只钉 layout 没有把 cast 带偏。
func.func @reduce_sum_2d_ra(%dst: !ascendc.local_tensor<1x8xf32>, %src: !ascendc.local_tensor<4x8xf32>,
                            %tmp: !ascendc.local_tensor<?xui8>, %r: i32, %a: i32) {
  ascendc.reduce_sum_2d_l2 %dst, %src, %tmp, %r, %a {layout = 1 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<1x8xf32>, !ascendc.local_tensor<4x8xf32>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}

// CHECK-LABEL: reduce_sum_2d_ra
// CHECK: uint32_t {{[a-zA-Z0-9_]+}}_reduce_shape[] = {static_cast<uint32_t>({{[a-zA-Z0-9_]+}}), static_cast<uint32_t>({{[a-zA-Z0-9_]+}})};
// CHECK: ReduceSum<float, AscendC::Pattern::Reduce::RA, false>
