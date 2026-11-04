// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate \
// RUN:     --afir-pipeline-sync --target=Ascend910B1 %s | FileCheck %s --check-prefix=SYNC
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s \
// RUN:   | FileCheck %s --implicit-check-not=afir.pipe_barrier

// 同步族的第一个生产者：Vector -> Vector 的顺序由 `AFIRPipelineSync` 产出 **AFIR** 的
// `pipe_barrier`，再由同步族的 lowering 落到 `ascendc.pipe_barrier`。
//
// 此前这条顺序是直接写成后端算子的——AFIR 层看不见，也就无从校验（3.2：放弃 TQue 自动
// 同步之后正确性归 AFIR 自己）。队列路径（enque / deque 与释放屏障）仍产 AscendC 算子：
// TQue 的入队/出队自带隐式同步，叠一层显式 token 就是双重同步，那一段要与退出 TQue
// （ADR-11/12 的 pool + memref.view）一起改。
//
// **这一步不改变竞争检查的结论**，也不应该改变：`pipe_barrier` 是单 pipe 内的全序点，
// 而检查器已经假定同 pipe 程序序即执行序（同一条理由让 `token<from = X, to = X>` 被
// verifier 拒掉）。它的价值在于把同步从「conversion 内部顺手做的事」变成流水线里
// selection 之前的一步，这是跨 pipe 那些**真带边**的情形接下来能搬上来的结构前提。
//
// 用例必须是连续两次 Vector 计算：load -> add -> store 走不到 V -> V 这条分支，
// 现有的 afir-block-id-pipeline 覆盖不到它。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#output_desc = #afir.tensor_desc<axis = [0], role = <output>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
#load_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>
#second_queue = #afir.buffer_desc<space = <ub>, buffer_id = 4, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>
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
      axes = [
        <id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>,
        <id = 1, name = "zb", axis_type = BlockOuter, size = #afir.expr<"1">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 2>,
        <id = 2, name = "zi", axis_type = BlockInner, size = #afir.expr<"16">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 1>],
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
      afir.sched = #afir.sched<axis = [1, 2], loop_axis = 2>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed = afir.add %loaded, %loaded outs(%compute_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#compute_desc, #compute_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [1, 2], loop_axis = 2>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    %second_tile = memref.alloc() {afir.buffer_desc = #second_queue} : memref<16xf16>
    %twice = afir.add %computed, %computed outs(%second_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#compute_desc, #compute_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [1, 2], loop_axis = 2>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %twice outs(%output) {
      afir.destination_descs = [#output_desc],
      afir.operand_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [1, 2], loop_axis = 2>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}


// SYNC-LABEL: func.func private @stage0_template0
// 两次 add 之间是 V -> V，产出 AFIR 层的屏障。
// SYNC: afir.pipe_barrier vector

// CHECK-LABEL: module @plan0
// CHECK: func.func private @stage0_template0
// CHECK: ascendc.pipe_barrier pipe_v
// CHECK: ascendc.pipe_barrier pipe_v
