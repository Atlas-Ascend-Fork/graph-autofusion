// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-gen-kernel-entry --verify-diagnostics --split-input-file %s

// tiling 上没有这块 workspace 的字段。这里必须报错而不是退化成偏移 0：退化会让两块
// workspace 叠在同一个地址上写，是静默的数据损坏。
module @plan0 attributes {
  afir.module_kind = "plan",
  afir.workspace_tensor_ids = array<i64: 2>,
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>,
  ascendc.tiling_schema = !emitasc.py_struct<"plan0TilingData", [
       !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>],
      ["graph0_result0_g0_tiling_data"]>
} {
  // expected-error @below {{GenKernelEntry: tiling data has no 'workspace2', so the offset of workspace 2 would never be assigned; ATT did not declare this workspace}}
  func.func private @stage0_template0(
       %workspace: memref<ui8, #afir.space<gm>> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
       %offset2: ui64 {afir.kernel_arg = #afir.kernel_arg<role = workspace_offset, index = 2>},
       %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
        ascendc.num_inputs = 0 : i32,
        afplan.variant = #afplan.variant<stage = 0, template = 0>
      } {
    return
  }
}
