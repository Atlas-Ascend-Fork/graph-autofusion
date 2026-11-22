// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 检查 reduce_sum_2d_l2 生成的 C++，尤其是运行期 srcShape 数组的元素转换。
//
// srcShape 的 ODS 类型为 Variadic<I32>，signless i32 默认生成为 int32_t，
// 但 Reduce API 要求 const uint32_t[]。初始化数组时必须显式转换每一项，
// 否则 C++11 列表初始化将运行期 signed -> unsigned 视为窄化，bisheng 报错。
//
// 本用例直接检查生成的文本；主机编译和使用其他重载的 CPU E2E 不会覆盖该 device C++ 错误。
//
// 直接输入 AscendC op，以隔离 printer 的契约。完整 plan 流水线还含 afplan 属性和
// index.ceildivu，ascir-translate 未注册这两个方言，不能用于这个生成测试。

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
