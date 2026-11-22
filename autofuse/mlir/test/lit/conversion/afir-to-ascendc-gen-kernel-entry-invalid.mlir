// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-gen-kernel-entry --verify-diagnostics --split-input-file %s

// expected-error @+1 {{GenKernelEntry: missing Plan ascendc.tiling_schema}}
module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
}

// -----

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>,
  ascendc.tiling_schema = !emitasc.py_struct<"Plan0TilingData", [
      !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>], ["graph0_result0_g0_tiling_data"]>
} {
  // 两个参数都自称「第 0 个输入」。这条现在在解析期就被方言 verifier 挡住，不再
  // 走到 GenKernelEntry 的同名检查——门禁前移了，诊断也更靠近错误本身。
  //
  // GenKernelEntry 里那条判重保留：按 5.3 的注记，`genVerifyDecl` 不是全路径门禁，
  // 程序化构造（`Attr::get` + `setArgAttr`）绕不过这里、却绕得过属性 verifier。两道
  // 门禁分管文本与构造两条路径。
  // expected-error @+1 {{duplicate kernel_arg role 'input' at argument 1; argument 0 already declares it}}
  func.func private @stage0_template0(
      %input0: memref<1xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>},
      %input1: memref<1xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>},
      %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
          <id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
        ascendc.num_inputs = 1 : i32,
        afplan.variant = #afplan.variant<stage = 0, template = 0>
      } {
    return
  }
}

// -----

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>,
  ascendc.tiling_schema = !emitasc.py_struct<"Plan0TilingData", [
      !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>], ["graph0_result0_g0_tiling_data"]>
} {
  // expected-error @+1 {{GenKernelEntry: template requires a non-negative afir.asc_graph tiling_key}}
  func.func private @stage0_template0(
      %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = -1, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
        ascendc.num_inputs = 0 : i32,
        afplan.variant = #afplan.variant<stage = 0, template = 0>
      } {
    return
  }
}

// -----

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [2]>,
  ascendc.tiling_schema = !emitasc.py_struct<"Plan0TilingData", [
      !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>], ["graph0_result0_g0_tiling_data"]>
} {
  func.func private @stage0_template0(
      %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 10, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
        ascendc.num_inputs = 0 : i32,
        afplan.variant = #afplan.variant<stage = 0, template = 0>
      } {
    return
  }
  // expected-error @+1 {{GenKernelEntry: duplicate tiling_key in stage 0}}
  func.func private @stage0_template1(
      %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 10, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
         ascendc.num_inputs = 0 : i32,
         afplan.variant = #afplan.variant<stage = 0, template = 1>
      } {
    return
  }
}

// -----

// expected-error @+1 {{GenKernelEntry: Cube/CV Plan is not supported}}
module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afir.cube_type = 1 : i64,
  afplan.stages = #afplan.stages<template_counts = [1]>,
  ascendc.tiling_schema = !emitasc.py_struct<"Plan0TilingData", [
      !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>], ["graph0_result0_g0_tiling_data"]>
} {
  func.func private @stage0_template0(
      %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    return
  }
}

// -----

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>,
  ascendc.tiling_schema = !emitasc.py_struct<"Plan0TilingData", [
      !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>], ["graph0_result0_g0_tiling_data"]>
} {
  // expected-error @+1 {{GenKernelEntry: invalid template coordinates}}
  func.func private @stage0_template1(
      %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {afplan.variant = #afplan.variant<stage = 1, template = 1>} {
    return
  }
}

// -----

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>,
  ascendc.tiling_schema = !emitasc.py_struct<"Plan0TilingData", [
      !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>], ["graph0_result0_g0_tiling_data"]>
} {
  // expected-error @+1 {{GenKernelEntry: template must contain exactly one stage tiling argument}}
  func.func private @stage0_template0(
      %tiling: !emitasc.py_struct<"WrongStageTilingData", [ui32], ["tiling_key"]>)
      attributes {afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    return
  }
}

// -----

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>,
  ascendc.tiling_schema = !emitasc.py_struct<"Plan0TilingData", [
      !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>], ["graph0_result0_g0_tiling_data"]>
} {
  // expected-error @+1 {{GenKernelEntry: ascendc.need_sync_all must be a unit attribute}}
  func.func private @stage0_template0(
      %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {ascendc.need_sync_all = true, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    return
  }
}
