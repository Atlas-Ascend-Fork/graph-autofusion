// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.broadcast` 的**常量源**一档，reg 侧。与同目录的 `afir-broadcast-duplicate.mlir`
// 是同一个 op、同一个判据、同一个调用——reg 侧的 `BroadcastRegApiCall::Generate` 在
// `IsBroadcastConstantTensor` 为真时同样调 `BroadcastScalar`。
//
// 两份用例唯一的差别是那块 tmp buffer：mem 侧的调度给一块
// （`BroadcastAscIrCodegenImpl::CalcTmpBufSize`），reg 侧不给（`...ImplV2` 没有这个覆写），
// 而 `checkTmpBuffer` 拿块数严格对等。这就是同一个调用要分成两个 implementation 的
// 全部理由，与芯片无关。下面那条 WRONGSIDE 反过来钉住这件事。
//
// 输入写成 memref 形态、跑整条流水，不是 `!ascendc.local_tensor` 的简写形态：
// 这一档要生成一对标量读事件，而事件号的持有者是资源物化阶段建的那个 `ascendc.pipe`，
// 源在哪条管线上也要看它落在队列还是 TBuf 上——两样都只有跑过 `--afir-memory-allocate`
// 才存在。写成简写形态的话这条用例根本走不到要钉的那段。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend950PR_9599 %s | FileCheck %s --check-prefix=IR
// kernel 文本由 `--afir-aggregate-launch` 直接出（它自己翻译，不再过 `ascir-translate`）。
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend950PR_9599 %s -o %t.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=broadcast_duplicate_reg_base %t.mlir \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --target=Ascend950PR_9599 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT
// 不带 tmp 的这一份在 mem 侧选不出实现——mem 侧的调度一定会给那一块。
// RUN: not af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync --afir-select-implementation %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WRONGSIDE

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
    %broadcast = afir.broadcast %loaded outs(%dst) {
      afir.destination_descs = [#dst_tile],
      afir.operand_descs = [#src_tile],
      afir.result_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>
    } : memref<1x1x1xf32>, memref<?x?x?xf32> -> memref<?x?x?xf32>
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
// 事件对在标量读之前：`Duplicate` 从 S 管线取这个值，而队列的 DeQue 只给 MTE2→V。
// IR: ascendc.pipe.fetch_event_id {{.*}}mte2_s
// IR: ascendc.set_flag {{.*}}mte2_s
// IR: ascendc.wait_flag {{.*}}mte2_s
// IR: ascendc.local_tensor.get_value
// IR: ascendc.duplicate_l2
// 三轴 / 四轴那两条分支这一档接不了，生成就是判据放宽了。
// IR-NOT: ascendc.broadcast_l2
// IR-NOT: afir.broadcast

// CPP: AscendC::HardEvent::MTE2_S
// CPP: .GetValue(
// 标准调用，不是 `duplicate.h` 的包装：那份包装只覆盖 8 位与 64 位整数。
// CPP: AscendC::Duplicate(
// CPP-NOT: Broadcast

// SELECT: ascend.implementation = {id = "broadcast.tensor.duplicate_reg_base", kind = "ascendc_api"}

// WRONGSIDE: no supported
