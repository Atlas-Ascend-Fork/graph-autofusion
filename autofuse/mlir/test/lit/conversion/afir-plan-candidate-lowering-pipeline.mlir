// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s

// Check one candidate module from ScheduleToAfir::PlanToAfir: one stage,
// one template and a load -> add -> store body. The pipeline prepares the body
// signature, lowers AFIR to AscendC, then generates tiling data and entry dispatch.

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#output_desc = #afir.tensor_desc<axis = [0], role = <output>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
#load_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>
#output_queue = #afir.buffer_desc<space = <ub>, buffer_id = 3, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(%input: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.buffer_desc = #gm_buffer}, %output: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#input_desc, #output_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 10,
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %load_tile = memref.alloc() {afir.buffer_desc = #load_buffer} : memref<16xf16>
    %compute_tile = memref.alloc() {afir.buffer_desc = #output_queue} : memref<16xf16>
    %loaded = afir.load %input outs(%load_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#input_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed = afir.add %loaded, %loaded outs(%compute_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#compute_desc, #compute_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %computed outs(%output) {
      afir.destination_descs = [#output_desc],
      afir.operand_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// CHECK-LABEL: module @plan0
// CHECK-SAME: ascendc.tiling_schema
// CHECK: func.func private @stage0_template0
// CHECK-SAME: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData"
// CHECK: ascendc.add_l2
// CHECK-NOT: afir.add
// CHECK-NOT: afir.load
// CHECK-NOT: afir.store
// CHECK: func.func @plan0_main
// CHECK-SAME: #afir.kernel_arg<role = tiling, index = 0>
// stage 的 tiling 就是 `_main` 的形参，不再从 plan 级结构体上取成员。
// CHECK-NOT: emitasc.member {{.*}} "graph0_result0_tiling_data"
// CHECK: emitasc.member {{.*}} "tiling_key"
// CHECK: scf.index_switch
// CHECK: case 10
// CHECK: func.call @stage0_template0
