// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.broadcast` 的**标量源**一档。与 `afir-broadcast-duplicate.mlir` 是同一个 tier
// 的另一种源形态：那边源是一块 tile（legacy `BroadcastScalar` 的第三支，事件 +
// `GetValue(0)`），这边源就是一个裸标量 SSA 值（第一支 `in.is_constant`，
// `codegen/api_call/broadcast/broadcast_api_call.cpp:249` 把常量名直接拼进调用）。
//
// bridge 对 ASCIR `Scalar` / `IndexExpr` 节点「不建 op 只建值」，给出的正是这种裸标量；
// `mean` 被拆成 `sum` + `truediv` 后，除数那一路就长这样。
//
// 这份用例真正在钉的是**没有**生成什么：没有标量读、没有读值前的同步事件、源上不取块内
// 下标。少写一条 CHECK-NOT，把标量当 tile 读的写法也能全绿。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=broadcast_duplicate_scalar %t.mlir \
// RUN:   | FileCheck %s --check-prefix=CPP
// 选出来的必须是标量源那一档。`--implicit-check-not` 钉的是互斥：tile 源那一档收到裸标量
// 要答不支持，它要是也收下了，这里就会出现两个 id 里的另一个。
// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT --implicit-check-not=broadcast.tensor.duplicate

// 目的地跨步时这一档照旧接不了：那道闸查的是目的地，与源是 tile 还是标量无关。
// RUN: sed 's/vectorized_strides = \[#afir.expr<"1">\]/vectorized_strides = [#afir.expr<"2">]/' %s > %t.strided.mlir
// RUN: not af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %t.strided.mlir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STRIDED
// STRIDED: 'afir.broadcast' op has no supported Ascend implementation

#gm_out = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">], role = <output>>
// 标量源在描述子列表里占一格，但不描述任何事实——它没有轴、没有缓冲、没有地址空间
// （`verifyDescriptorList` 的 `allowsScalars` 一支）。占位是必须的，否则描述子下标就与
// 操作数下标错开了。
#scalar_src = #afir.tensor_desc<role = <compute>>
#dst_tile = #afir.tensor_desc<axis = [3, 4, 2],
                              repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                         #afir.expr<"zt_size">],
                              strides = [#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">,
                                         #afir.expr<"1">],
                              vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %output: memref<128xf32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>,
                                afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"512">,
                                                                     life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#gm_out],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"128">, split_pair_other_id = 0>,
              <id = 1, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((128 / (zt_size)))">,
               from = [0], split_pair_other_id = 2>,
              <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">, from = [0],
               split_pair_other_id = 1>,
              <id = 3, name = "zTB", axis_type = BlockOuter,
               size = #afir.expr<"Ceiling((Ceiling((128 / (zt_size))) / (zTb_size)))">, from = [1],
               split_pair_other_id = 4>,
              <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">, from = [1],
               split_pair_other_id = 3>],
      type = Compute,
      size_var = [#afir.expr<"zt_size">, #afir.expr<"zTb_size">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %tmp_bytes = afir.tiling_param "b0_size" : index
    %tile = afir.tiling_param "zt_size" : index
    %block = afir.tiling_param "zTb_size" : index
    %outer = afir.symbolic_eval %block, %tile {expr = #afir.expr<"(128 / (__arg0 * __arg1))">}
        : (index, index) -> index
    %scalar = arith.constant 2.000000e+00 : f32
    %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?xf32>
    %tmp = memref.alloc(%tmp_bytes) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                                          size_from_resource = true,
                                                                          alignment_bytes = 32,
                                                                          life_time_axis_id = -1>} : memref<?xui8>
    %broadcast = afir.broadcast %scalar outs(%dst) tmp(%tmp) {
      afir.destination_descs = [#dst_tile],
      afir.operand_descs = [#scalar_src],
      afir.result_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>
    } : f32, memref<?x?x?xf32>, memref<?xui8> -> memref<?x?x?xf32>
    afir.store %broadcast outs(%output) {
      afir.destination_descs = [#gm_out],
      afir.operand_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d3)>]
    } : memref<?x?x?xf32>, memref<128xf32>
    return
  }
}

// IR-LABEL: func.func private @stage0_template0
// IR: %[[SCALAR:.*]] = arith.constant 2.000000e+00 : f32
// 铺满的元素数仍取目的地那一段的跨度（带尾块修正），与 tile 源那一份逐字同一条路。
// IR: scf.for
// IR: %[[ACTUAL:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 2, role = actual_size>}
// IR: %[[REMAINDER:.*]] = arith.subi %[[ACTUAL]], %{{.*}} : index
// IR: %[[COUNT:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// 标量直接当实参：第二个操作数就是上面那个常量，不是从某块 tile 读出来的值。
// IR: ascendc.duplicate_l2 %{{.*}}, %[[SCALAR]], %[[COUNT]]
// 这三条是这份用例的要点：没有标量读、没有为标量读准备的事件对。
// IR-NOT: ascendc.local_tensor.get_value
// IR-NOT: mte2_s
// IR-NOT: ascendc.broadcast_l2
// IR-NOT: afir.broadcast

// CPP: AscendC::Duplicate(
// CPP-NOT: .GetValue(
// CPP-NOT: MTE2_S
// CPP-NOT: Broadcast

// SELECT: ascend.implementation = {id = "broadcast.scalar.duplicate", kind = "ascendc_api"}
