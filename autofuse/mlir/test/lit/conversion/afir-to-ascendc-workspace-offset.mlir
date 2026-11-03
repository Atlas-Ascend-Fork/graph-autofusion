// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-gen-kernel-entry %s | af-opt | FileCheck %s

// workspace 偏移链路的入口一侧：entry 从 stage tiling 上读 `workspace<ID>`，作为实参传给
// 声明了 `role = workspace_offset` 的那个形参。`index` 就是这块 workspace 的 buffer id，
// 因此两块 workspace 各取各的字段，不会串。
module @plan0 attributes {
  afir.module_kind = "plan",
  afir.workspace_tensor_ids = array<i64: 2, 5>,
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>,
  ascendc.tiling_schema = !emitasc.py_struct<"plan0TilingData", [
       !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui64, ui64, ui32],
                          ["workspace2", "workspace5", "tiling_key"]>],
      ["graph0_result0_g0_tiling_data"]>
} {
  func.func private @stage0_template0(
       %workspace: memref<ui8, #afir.space<gm>> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
       %offset2: ui64 {afir.kernel_arg = #afir.kernel_arg<role = workspace_offset, index = 2>},
       %offset5: ui64 {afir.kernel_arg = #afir.kernel_arg<role = workspace_offset, index = 5>},
       %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui64, ui64, ui32],
                                   ["workspace2", "workspace5", "tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 7, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
        ascendc.num_inputs = 0 : i32,
        afplan.variant = #afplan.variant<stage = 0, template = 0>
      } {
    return
  }
}

// CHECK-LABEL: module @plan0
// CHECK: func.func @plan0_main
// entry 自己的签名里只有 workspace，偏移不外露：它是从 tiling 里读出来的，不是 launch 传的。
// CHECK-NOT: role = workspace_offset
// stage 的 tiling 就是 `_main` 的形参（ATT 的最外层布局没有 plan 那一层），偏移从它的
// `workspace<ID>` 成员上读。
// CHECK: %[[STAGE:[a-z0-9_]+]]: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData"
// CHECK-SAME: #afir.kernel_arg<role = tiling, index = 0>
// CHECK: scf.index_switch
// CHECK: case 7
// CHECK-DAG: %[[OFF2:.*]] = emitasc.member %[[STAGE]] "workspace2"
// CHECK-DAG: %[[OFF5:.*]] = emitasc.member %[[STAGE]] "workspace5"
// CHECK: func.call @stage0_template0({{.*}}, %[[OFF2]], %[[OFF5]], %[[STAGE]])
