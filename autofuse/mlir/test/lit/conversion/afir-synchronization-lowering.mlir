// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate %s | FileCheck %s --check-prefix=LOOP
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --implicit-check-not=afir.event_ --implicit-check-not=afir.pipe_barrier

// 验证 token、event_record、event_wait 和 pipe_barrier 到 AscendC 的下降。
// token 按 ADR-19 携带有向 pipe 对，可直接映射到 HardEvent，无需重新推断方向。
//
// 事件槽按 legacy 的纪律管理：record 侧 `AllocEventID` 占位、wait 侧 `WaitFlag` 后
// `ReleaseEventID` 归还——交叠的同型对各自持有不同的槽，顺序的语义不会丢。
// DAG coloring 与 profile 槽位上限属于后续步骤（本夹具里 record/wait 相邻，槽位
// 分配了立刻归还）。

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
    // 两条方向相反的边同时在场：如果映射把 pipe 对当成无序集合，两条会塌成同一个
    // HardEvent，下面的 CHECK 就分不出来了。
    %in_token = afir.event_record : !afir.token<from = transfer_in, to = vector>
    afir.event_wait %in_token : !afir.token<from = transfer_in, to = vector>
    %out_token = afir.event_record : !afir.token<from = vector, to = transfer_out>
    afir.event_wait %out_token : !afir.token<from = vector, to = transfer_out>
    afir.pipe_barrier vector
    // 全序原语：全引擎屏障与跨核栅栏（hard 变体）。无操作数、无 token，落点由 Op 本身
    // 决定——PIPE_ALL 与 SyncAll<true>（非 cube 范围内参与的全是 AIV 核，AIC+AIV 混模
    // 协议的 `<false>` 在纯 AIV 生成路径下死锁）。
    afir.barrier_all
    afir.sync_all
    // 只检查 Cube 路径的同步骨架，不包含 Cube 计算：GM->L1（MTE2）、L1->L0A/L0B（MTE1）、
    // MAD（M）、L0C 搬出（FIX）及 GM 搬出（MTE3）。四条跨引擎边必须分别映射到正确 HardEvent，
    // 不能将各引擎合并视为 PIPE_V。
    %l1_token = afir.event_record : !afir.token<from = transfer_in, to = transfer_local>
    afir.event_wait %l1_token : !afir.token<from = transfer_in, to = transfer_local>
    %mad_token = afir.event_record : !afir.token<from = transfer_local, to = cube>
    afir.event_wait %mad_token : !afir.token<from = transfer_local, to = cube>
    %fix_token = afir.event_record : !afir.token<from = cube, to = fixpipe>
    afir.event_wait %fix_token : !afir.token<from = cube, to = fixpipe>
    %drain_token = afir.event_record : !afir.token<from = fixpipe, to = transfer_out>
    afir.event_wait %drain_token : !afir.token<from = fixpipe, to = transfer_out>
    afir.pipe_barrier cube
    afir.store %computed outs(%output) {
      afir.destination_descs = [#output_desc],
      afir.operand_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [1, 2], loop_axis = 2>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// LOOP-LABEL: func.func private @stage0_template0
// 同步 op 虽不带 sched，循环生成仍须保留其与相邻算子的顺序，不能移到循环外。
// LOOP: scf.for
// LOOP: afir.add
// LOOP: afir.event_record
// LOOP: afir.event_wait
// LOOP: afir.pipe_barrier
// LOOP: afir.barrier_all
// LOOP: afir.sync_all
// LOOP: afir.store

// CHECK-LABEL: module @plan0
// CHECK: func.func private @stage0_template0
// transfer_in -> compute 落到 mte2_v；下面那条方向相反的落到 v_mte3。
// CHECK: %[[IN:.*]] = ascendc.pipe.alloc_event_id %{{.*}}, mte2_v : i8
// CHECK: ascendc.set_flag mte2_v, %[[IN]] : i8
// CHECK: ascendc.wait_flag mte2_v, %[[IN]] : i8
// CHECK: ascendc.pipe.release_event_id
// CHECK: %[[OUT:.*]] = ascendc.pipe.alloc_event_id %{{.*}}, v_mte3 : i8
// CHECK: ascendc.set_flag v_mte3, %[[OUT]] : i8
// CHECK: ascendc.wait_flag v_mte3, %[[OUT]] : i8
// CHECK: ascendc.pipe.release_event_id
// CHECK: ascendc.pipe_barrier pipe_v
// 检查全引擎屏障与 hard 跨核栅栏。后者显式携带 isAIVOnly，自动 emitter 输出
// 默认 true 的 SyncAll()。纯 AIV 范围不能使用等待 AIC 参与的 false 协议，否则会死锁。
// CHECK: ascendc.pipe_barrier pipe_all
// CHECK: ascendc.sync_all_hard {isAIVOnly}
// 检查 Cube 同步链的四个 HardEvent，分别保留 transfer_local、cube、fixpipe 等引擎身份。
// CHECK: %[[L1:.*]] = ascendc.pipe.alloc_event_id %{{.*}}, mte2_mte1 : i8
// CHECK: ascendc.set_flag mte2_mte1, %[[L1]] : i8
// CHECK: ascendc.wait_flag mte2_mte1, %[[L1]] : i8
// CHECK: ascendc.pipe.release_event_id
// CHECK: %[[MAD:.*]] = ascendc.pipe.alloc_event_id %{{.*}}, mte1_m : i8
// CHECK: ascendc.set_flag mte1_m, %[[MAD]] : i8
// CHECK: ascendc.wait_flag mte1_m, %[[MAD]] : i8
// CHECK: ascendc.pipe.release_event_id
// CHECK: %[[FIX:.*]] = ascendc.pipe.alloc_event_id %{{.*}}, m_fix : i8
// CHECK: ascendc.set_flag m_fix, %[[FIX]] : i8
// CHECK: ascendc.wait_flag m_fix, %[[FIX]] : i8
// CHECK: ascendc.pipe.release_event_id
// CHECK: %[[DRAIN:.*]] = ascendc.pipe.alloc_event_id %{{.*}}, fix_mte3 : i8
// CHECK: ascendc.set_flag fix_mte3, %[[DRAIN]] : i8
// CHECK: ascendc.wait_flag fix_mte3, %[[DRAIN]] : i8
// CHECK: ascendc.pipe.release_event_id
// 各引擎的 pipe_barrier 也必须保留对应 pipe，不能统一生成为 PIPE_V。
// CHECK: ascendc.pipe_barrier pipe_m
