// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.broadcast` 的**常量源**一档，mem 侧。源在它唯一那根向量化轴上 repeat 是 1，
// legacy 这时发的不是 `Broadcast` 而是 `Duplicate`（`broadcast_api_call.cpp` 的
// `IsBroadcastConstantTensor` → `BroadcastScalar`）。
//
// **输入写成 memref 形态、跑整条流水**，不是 `!ascendc.local_tensor` 的简写形态：
// 这一档要发一对标量读事件，而事件号的持有者是资源物化阶段建的那个 `ascendc.pipe`，
// 源在哪条管线上也要看它落在队列还是 TBuf 上——两样都只有跑过 `--afir-memory-allocate`
// 才存在。写成简写形态的话这条用例根本走不到要钉的那段。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// kernel 文本由 `--afir-aggregate-launch` 直接出（它自己翻译，不再过 `ascir-translate`）。
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=broadcast_duplicate %t.mlir \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT
// mem 侧的调度给一块 tmp buffer，reg 侧不给（`BroadcastAscIrCodegenImplV2` 没有
// `CalcTmpBufSize` 覆写）。所以带 tmp 的这一份在 reg 侧选不出实现——两侧分成两个
// implementation 的理由就是这个，不是芯片。
// RUN: not af-opt --target=Ascend950PR_9599 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync --afir-select-implementation %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WRONGSIDE

// A strided destination cannot be filled by a contiguous Duplicate over its span.
// Keep source/result bindings coherent; only the destination layout changes.
// RUN: sed 's/vectorized_strides = \[#afir.expr<"1">\]/vectorized_strides = [#afir.expr<"2">]/' %s > %t.strided.mlir
// RUN: not af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %t.strided.mlir 2>&1 | FileCheck %s --check-prefix=STRIDED
// STRIDED: 'afir.broadcast' op has no supported Ascend implementation

#gm_in = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], role = <input>>
#gm_out = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">], role = <output>>
// 源 tile：向量化轴（2 号，TileInner）上 repeat 1、stride 0——「每个迭代读同一份」，
// 也就是这一档的判据。
#src_tile = #afir.tensor_desc<axis = [3, 4, 2],
                              repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">],
                              strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">],
                              vectorized_axis = [2], vectorized_strides = [#afir.expr<"0">], role = <compute>>
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
      %input: memref<1xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                             afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4">,
                                                                  life_time_axis_id = -1>},
      %output: memref<128xf32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>,
                                afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"512">,
                                                                     life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#gm_in, #gm_out],
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
    %src = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                                                slot_count = 1, queue_depth = 2,
                                                                size_from_resource = true,
                                                                life_time_axis_id = -1>} : memref<1x1x1xf32>
    %loaded = afir.load %input outs(%src) {
      afir.destination_descs = [#src_tile],
      afir.operand_descs = [#gm_in],
      afir.result_descs = [#src_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>]
    } : memref<1xf32>, memref<1x1x1xf32> -> memref<1x1x1xf32>
    %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?xf32>
    %tmp = memref.alloc(%tmp_bytes) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                                          size_from_resource = true,
                                                                          alignment_bytes = 32,
                                                                          life_time_axis_id = -1>} : memref<?xui8>
    %broadcast = afir.broadcast %loaded outs(%dst) tmp(%tmp) {
      afir.destination_descs = [#dst_tile],
      afir.operand_descs = [#src_tile],
      afir.result_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>
    } : memref<1x1x1xf32>, memref<?x?x?xf32>, memref<?xui8> -> memref<?x?x?xf32>
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
// 铺满的元素数是**目的地**那一段的长度（带尾块修正），不是源那一侧的。
// 源在向量化轴上 stride 为 0，按源取会铺成 1 个元素、静默少写一整块——这个 op 有两个
// 张量，分段结果是共享的，认错列不会报错、只会少写。
// 先锚到 tile 循环上：循环之前还有几条形状相同的三元式（block 维的尾块）。
// IR: scf.for
// IR: %[[ACTUAL:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 2, role = actual_size>}
// IR: %[[REMAINDER:.*]] = arith.subi %[[ACTUAL]], %{{.*}} : index
// IR: %[[COUNT:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// 事件对在标量读**之前**：`Duplicate` 从 S 管线取这个值，而队列的 DeQue 只给 MTE2→V。
// IR: ascendc.pipe.fetch_event_id {{.*}}mte2_s
// IR: ascendc.set_flag {{.*}}mte2_s
// IR: ascendc.wait_flag {{.*}}mte2_s
// IR: ascendc.local_tensor.get_value
// IR: ascendc.duplicate_l2 %{{.*}}, %{{.*}}, %[[COUNT]]
// 三轴 / 四轴那两条分支这一档接不了，发出来就是判据放宽了。
// IR-NOT: ascendc.broadcast_l2
// IR-NOT: afir.broadcast

// CPP: AscendC::HardEvent::MTE2_S
// CPP: .GetValue(
// 标准调用，不是 `duplicate.h` 的包装：那份包装只覆盖 8 位与 64 位整数。
// CPP: AscendC::Duplicate(
// CPP-NOT: Broadcast

// SELECT: ascend.implementation = {id = "broadcast.tensor.duplicate", kind = "ascendc_api"}

// WRONGSIDE: no supported
