// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `srcShape` 在 ODS 里是 `Variadic<I32>`，但 API 按下标 0 / 1 读死两项
// （`reduce.h:233` 的 `const uint32_t srcShape[]`）。printer 把操作数原样展开成栈数组，
// 于是给一项就编出一个一元数组、API 仍读两次——**运行期越界读，编译期什么都看不出来**。
//
// 这个 op 的另外三处不规则（type 模板参数、`isReuseSource`、栈数组）都由手写 printer 兜着，
// 唯独 arity 兜不了：printer 只负责把拿到的东西写出去，它无从判断拿到的对不对。所以这条
// 契约必须落在 verifier 上。

// RUN: af-opt --verify-diagnostics --split-input-file %s

func.func @src_shape_too_few(%dst: !ascendc.local_tensor<4x1xf32>, %src: !ascendc.local_tensor<4x8xf32>,
                             %tmp: !ascendc.local_tensor<?xui8>, %a: i32) {
  // expected-error @+1 {{expects exactly 2 srcShape extents, got 1}}
  ascendc.reduce_sum_2d_l2 %dst, %src, %tmp, %a {layout = 0 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xf32>, !ascendc.local_tensor<4x8xf32>, !ascendc.local_tensor<?xui8>, i32
  return
}

// -----

func.func @src_shape_too_many(%dst: !ascendc.local_tensor<4x1xf32>, %src: !ascendc.local_tensor<4x8xf32>,
                              %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %b: i32, %c: i32) {
  // expected-error @+1 {{expects exactly 2 srcShape extents, got 3}}
  ascendc.reduce_sum_2d_l2 %dst, %src, %tmp, %a, %b, %c {layout = 0 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xf32>, !ascendc.local_tensor<4x8xf32>, !ascendc.local_tensor<?xui8>, i32, i32, i32
  return
}
