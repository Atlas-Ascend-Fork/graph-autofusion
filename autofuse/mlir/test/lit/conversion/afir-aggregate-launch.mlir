// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.lowered.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=fused_graph %t.lowered.mlir | FileCheck %s

// This fixture chains addPlanCandidateLoweringPipeline's per-candidate lowering
// (same single-candidate load -> add -> store plan as
// afir-plan-candidate-lowering-pipeline.mlir, plus afir.subgraph_id which
// GenAggregateTilingData's Plan contract requires) with
// runAggregateLaunchDriver's aggregate + AutofuseLaunch text emission, driven
// entirely through af-opt so the whole per-candidate-lowering ->
// aggregate-launch path has ST coverage independent of the C++ codegen
// orchestration entry point.


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

// CHECK: // -----KERNEL_BODY-----
// CHECK: #include <cassert>
// CHECK: #include "autofuse_tiling_data.h"
// CHECK-NOT: #include "TilingData.h"
// CHECK: __aicore__ void plan0_stage0_template0
// CHECK: assert(
// CHECK: AscendC::Add
// CHECK: __aicore__ void plan0_main
// CHECK: extern "C"  __global__ __aicore__ void fused_graph
// CHECK: // -----TILING_DATA-----
// 只发两个结构体：这个 schedule group 自己的，和最外层的。plan 级的那张 schema 只是
// 载体（`GenTilingData`），它不是根的字段，因此不进发射递归——ATT 的最外层布局里也没有
// plan 这一层。
// CHECK: struct AscGraph0ScheduleResult0G0TilingData
// CHECK-NOT: struct plan0TilingData
// CHECK: struct AutofuseTilingData
// CHECK: // -----AUTOFUSE_LAUNCH-----
// CHECK: extern "C" int64_t AutofuseLaunch
// CHECK: fused_graph<<<blockDim, nullptr, stream>>>
