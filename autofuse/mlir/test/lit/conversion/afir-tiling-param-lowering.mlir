// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-kernel-body-signature --afir-materialize-tiling-params %s | FileCheck %s --check-prefix=AFIR --implicit-check-not=emitasc.member
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --implicit-check-not=afir.tiling_param
// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate --afir-verify-kernel-complete %s | FileCheck %s --check-prefix=AFIR --implicit-check-not=emitasc.member

// 尺寸的 SSA 化两端各验一次。
//
// AFIR 端：运行期尺寸是 `afir.tiling_param`，函数体里没有 `emitasc.member`。第三条
// RUN 把这件事交给判据自己说——`--afir-verify-kernel-complete` 默认就跑 C3（函数体里
// 只出现 AFIR 与它复用的标准方言），循环生成之后仍然通过。此前循环生成当场建一个
// `emitasc.member`，同一条命令会在那里失败。
//
// AscendC 端：同一个名字变回 `tiling.q2_size`。**host ABI 逐字节不变**——下面
// CHECK 的 schema 字段表与迁移前逐字相同，换的只是 kernel 侧谁在什么时候建这个访问。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#output_desc = #afir.tensor_desc<axis = [0], role = <output>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
// 队列尺寸标 `size_from_resource`，字段名不写进 IR：`q2_size` 由 `buffer_id = 2` 加
// 「它是队列」派生出来，两端（TilingData 建字段 / kernel 侧取符号）用的是同一处拼法。
#load_queue = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>
#output_queue = #afir.buffer_desc<space = <ub>, buffer_id = 3, slot_count = 2, queue_depth = 2, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>

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
    // `GenTilingData` 按这份 id 表往 stage schema 里补 `q{id}_size`，host 侧的字段
    // 因此与 kernel 侧取的名字是同一批。bridge 建图时写的就是这个属性。
    afir.queue_ids = array<i64: 2, 3>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %load_tile = memref.alloc() {afir.buffer_desc = #load_queue} : memref<16xf16>
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

// 三个符号各一份，顺序与收集顺序一致：先轴尺寸，再各缓冲的字节数。
// AFIR-LABEL: func.func private @stage0_template0
// AFIR-SAME: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64], ["block_dim", "s0", "q2_size", "q3_size"]>
// 不钉 `: index`：循环生成会在 "s0" 这一份上补 `#afir.axis_binding`（它就是那根轴的
// 尺寸），第三条 RUN 走的正是那之后的形态。类型由 smoke 的 round-trip 用例钉住。
// AFIR: afir.tiling_param "s0"
// AFIR: afir.tiling_param "q2_size"
// AFIR: afir.tiling_param "q3_size"

// CHECK-LABEL: module @plan0
// CHECK: func.func private @stage0_template0
// CHECK-SAME: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData"
// CHECK-SAME: "s0", "q2_size", "q3_size"
// CHECK-DAG: emitasc.member %{{.*}} "s0"
// CHECK-DAG: emitasc.member %{{.*}} "q2_size"
// CHECK-DAG: emitasc.member %{{.*}} "q3_size"
