// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 同一 stage 的备选 template 使用不同 rank 的同身份 GM 参数。
// _main 统一 ABI，调用点恢复各自视图；template 保留 afir.arg_descs 所需的 ranked 签名。
// RUN: af-opt --afir-gen-kernel-entry %s | af-opt | FileCheck %s
// RUN: af-opt --afir-gen-kernel-entry %s | af-opt | FileCheck %s --check-prefix=TEMPLATE

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [2]>,
  ascendc.tiling_schema = !emitasc.py_struct<"plan0TilingData", [
       !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>],
      ["graph0_result0_g0_tiling_data"]>
} {
  func.func private @stage0_template0(
       %out: memref<4xf32, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>},
       %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 10, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
         ascendc.num_inputs = 0 : i32,
         afplan.variant = #afplan.variant<stage = 0, template = 0>
      } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    memref.store %zero, %out[%c0] : memref<4xf32, 22 : i32>
    return
  }
  func.func private @stage0_template1(
       %out: memref<1x4xf32, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>},
       %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 20, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
         ascendc.num_inputs = 0 : i32,
         afplan.variant = #afplan.variant<stage = 0, template = 1>
      } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    memref.store %zero, %out[%c0, %c0] : memref<1x4xf32, 22 : i32>
    return
  }
}

// 一张实参表，一个 ABI 类型。
// CHECK-LABEL: module @plan0
// CHECK: func.func @plan0_main
// CHECK-SAME: %[[OUT:[A-Za-z0-9_]+]]: memref<*xf32, 22 : i32>
// CHECK-SAME: #afir.kernel_arg<role = output, index = 0>
// 每个备选在自己的调用点转回自己声明的视图。
// CHECK: scf.index_switch
// CHECK: %[[V0:[A-Za-z0-9_]+]] = memref.cast %[[OUT]] : memref<*xf32, 22 : i32> to memref<4xf32, 22 : i32>
// CHECK: func.call @stage0_template0(%[[V0]]
// CHECK: %[[V1:[A-Za-z0-9_]+]] = memref.cast %[[OUT]] : memref<*xf32, 22 : i32> to memref<1x4xf32, 22 : i32>
// CHECK: func.call @stage0_template1(%[[V1]]

// template 的签名一动不动——`afir.arg_descs` 把它的秩绑住了。单独一条 RUN 行查：两个 template 的
// 定义排在 `_main` 之前，而 `CHECK-DAG` 不回退到已扫过的位置。
// TEMPLATE-DAG: func.func private @stage0_template0(%{{[A-Za-z0-9_]+}}: memref<4xf32, 22 : i32>
// TEMPLATE-DAG: func.func private @stage0_template1(%{{[A-Za-z0-9_]+}}: memref<1x4xf32, 22 : i32>
