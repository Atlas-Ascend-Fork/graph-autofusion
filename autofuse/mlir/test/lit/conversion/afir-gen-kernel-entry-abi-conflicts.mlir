// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 视图差异可以统一；同身份参数的元素类型或地址空间冲突必须拒绝。
// RUN: af-opt --afir-gen-kernel-entry --split-input-file --verify-diagnostics %s

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
  // expected-error @below {{GenKernelEntry: incompatible ABI types for output:0 across templates: 'memref<*xf32, 22 : i32>' and 'memref<*xf16, 22 : i32>'}}
  func.func private @stage0_template1(
       %out: memref<1x4xf16, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>},
       %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 20, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
         ascendc.num_inputs = 0 : i32,
         afplan.variant = #afplan.variant<stage = 0, template = 1>
      } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f16
    memref.store %zero, %out[%c0, %c0] : memref<1x4xf16, 22 : i32>
    return
  }
}

// -----

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
  // expected-error @below {{GenKernelEntry: incompatible ABI types for output:0 across templates: 'memref<*xf32, 22 : i32>' and 'memref<*xf32, 21 : i32>'}}
  func.func private @stage0_template1(
       %out: memref<1x4xf32, 21 : i32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>},
       %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 20, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
         ascendc.num_inputs = 0 : i32,
         afplan.variant = #afplan.variant<stage = 0, template = 1>
      } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    memref.store %zero, %out[%c0, %c0] : memref<1x4xf32, 21 : i32>
    return
  }
}
