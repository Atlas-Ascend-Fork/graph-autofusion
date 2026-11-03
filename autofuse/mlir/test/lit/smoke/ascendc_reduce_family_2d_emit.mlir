// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `ascendc.reduce_{max,min,prod,any,all}_2d_l2` 与 `ascendc.reduce_init` 新增的三个
// `reduceType` 发射到 C++ 文本这一层。
//
// 这一层是这一族**唯一**会出错的地方，理由与 `ascendc_reduce_sum_2d_emit.mlir` 那份逐字相同
// （`srcShape` 的 `static_cast<uint32_t>` 窄化），而且只有真机编 kernel 时才会撞上：主机侧
// 编译不碰 device 代码，CPU E2E 也不走这个重载。六个 op 共用同一个 printer 模板，所以窄化
// 那一条不会逐 op 各错一次；这份用例盯的是**函数名有没有跟着 op 换**——printer 取的是
// `getAPIName()`，接错了会把 `ReduceMin` 发成 `ReduceSum`，编得过、算错值。
//
// `reduce_init` 的三个新枚举同理：`kReduceOpAny` / `kReduceOpAll` / `kReduceOpMean` 是
// `ascendc/api/reduce_init.h` 里的全局常量，switch 里少一个分支或接错一个常量，填的就是
// 另一个单位元——全负输入上填 0 当最大值那一类静默错值。

// RUN: af-opt --convert-afir-to-ascendc --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s

func.func @reduce_max_2d(%dst: !ascendc.local_tensor<4x1xf32>, %src: !ascendc.local_tensor<4x8xf32>,
                         %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %r: i32) {
  ascendc.reduce_max_2d_l2 %dst, %src, %tmp, %a, %r {layout = 0 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xf32>, !ascendc.local_tensor<4x8xf32>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}

// CHECK-LABEL: reduce_max_2d
// CHECK: uint32_t {{[a-zA-Z0-9_]+}}_reduce_shape[] = {static_cast<uint32_t>({{[a-zA-Z0-9_]+}}), static_cast<uint32_t>({{[a-zA-Z0-9_]+}})};
// CHECK: ReduceMax<float, AscendC::Pattern::Reduce::AR, false>

// -----

func.func @reduce_min_2d(%dst: !ascendc.local_tensor<4x1xf32>, %src: !ascendc.local_tensor<4x8xf32>,
                         %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %r: i32) {
  ascendc.reduce_min_2d_l2 %dst, %src, %tmp, %a, %r {layout = 0 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xf32>, !ascendc.local_tensor<4x8xf32>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}

// CHECK-LABEL: reduce_min_2d
// CHECK: ReduceMin<float, AscendC::Pattern::Reduce::AR, false>

// -----

func.func @reduce_prod_2d(%dst: !ascendc.local_tensor<4x1xf32>, %src: !ascendc.local_tensor<4x8xf32>,
                          %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %r: i32) {
  ascendc.reduce_prod_2d_l2 %dst, %src, %tmp, %a, %r {layout = 0 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xf32>, !ascendc.local_tensor<4x8xf32>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}

// CHECK-LABEL: reduce_prod_2d
// CHECK: ReduceProd<float, AscendC::Pattern::Reduce::AR, false>

// -----

// RA 也过一遍：layout 在同一个 printer 里，接错了两种布局会一起错。
func.func @reduce_any_2d(%dst: !ascendc.local_tensor<1x8xf32>, %src: !ascendc.local_tensor<4x8xf32>,
                         %tmp: !ascendc.local_tensor<?xui8>, %r: i32, %a: i32) {
  ascendc.reduce_any_2d_l2 %dst, %src, %tmp, %r, %a {layout = 1 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<1x8xf32>, !ascendc.local_tensor<4x8xf32>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}

// CHECK-LABEL: reduce_any_2d
// CHECK: ReduceAny<float, AscendC::Pattern::Reduce::RA, false>

// -----

func.func @reduce_all_2d(%dst: !ascendc.local_tensor<4x1xf32>, %src: !ascendc.local_tensor<4x8xf32>,
                         %tmp: !ascendc.local_tensor<?xui8>, %a: i32, %r: i32) {
  ascendc.reduce_all_2d_l2 %dst, %src, %tmp, %a, %r {layout = 0 : i32, srcInnerPad = true}
      : !ascendc.local_tensor<4x1xf32>, !ascendc.local_tensor<4x8xf32>, !ascendc.local_tensor<?xui8>, i32, i32
  return
}

// CHECK-LABEL: reduce_all_2d
// CHECK: ReduceAll<float, AscendC::Pattern::Reduce::AR, false>

// -----

// `reduce_init` 新增的三个枚举。0..3（Min / Max / Sum / Prod）此前就有分支，这里只补新的。
func.func @reduce_init_new_types(%dst: !ascendc.local_tensor<4x8xf32>, %a: index, %r: index,
                                 %rcur: index, %inner: index, %stride: index) {
  ascendc.reduce_init %dst, %a, %r, %rcur, %inner, %stride {reduceType = 4 : i32, isTailLast = true}
      : !ascendc.local_tensor<4x8xf32>, index, index, index, index, index
  ascendc.reduce_init %dst, %a, %r, %rcur, %inner, %stride {reduceType = 5 : i32, isTailLast = true}
      : !ascendc.local_tensor<4x8xf32>, index, index, index, index, index
  ascendc.reduce_init %dst, %a, %r, %rcur, %inner, %stride {reduceType = 6 : i32, isTailLast = true}
      : !ascendc.local_tensor<4x8xf32>, index, index, index, index, index
  return
}

// CHECK-LABEL: reduce_init_new_types
// CHECK: AscendC::Duplicate
// CHECK: AscendC::Duplicate
// CHECK: AscendC::Duplicate
