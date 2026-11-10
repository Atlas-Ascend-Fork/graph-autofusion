// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// store 写 GM 后 load 回读（workspace 回读，MTE3 → MTE2）。回读 load 之前插入
// transfer_out→transfer_in 显式同步对。两个函数分别覆盖同值与别名（subview 追到同一
// 根 memref）两类夹具；第三个函数把回读者换成 `afir.nddma`——GM 写者判据必须是
// 方向 trait（TransferIn），按 op 名单点 load 的话 nddma 会静默漏掉这对同步。

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync %s | FileCheck %s --check-prefix=SYNC
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync %s | FileCheck %s --check-prefix=MULTI
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync %s | FileCheck %s --check-prefix=NDDMA

module {
  func.func private @store_load_workspace(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %workspace: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %output: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], role = <input>>,
      #afir.tensor_desc<axis = [0], role = <output>>,
      #afir.tensor_desc<axis = [0], role = <output>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %load_tile = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2,
                                                            slot_count = 2, queue_depth = 2,
                                                            size_bytes_expr = <"(2 * s0)">,
                                                            alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %compute_tile = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3,
                                                            slot_count = 2, queue_depth = 2,
                                                            size_bytes_expr = <"(2 * s0)">,
                                                            alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %input outs(%load_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed = afir.add %loaded, %loaded outs(%compute_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>,
                            #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %computed outs(%workspace) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    %back = afir.load %workspace outs(%load_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed2 = afir.add %back, %back outs(%compute_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>,
                            #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %computed2 outs(%output) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }

  func.func private @multi_writer_readback(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %workspace: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %output: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], role = <input>>,
      #afir.tensor_desc<axis = [0], role = <output>>,
      #afir.tensor_desc<axis = [0], role = <output>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %load_tile = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2,
                                                            slot_count = 2, queue_depth = 2,
                                                            size_bytes_expr = <"(2 * s0)">,
                                                            alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %compute_tile = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3,
                                                            slot_count = 2, queue_depth = 2,
                                                            size_bytes_expr = <"(2 * s0)">,
                                                            alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %input outs(%load_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed = afir.add %loaded, %loaded outs(%compute_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>,
                            #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %computed outs(%workspace) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    %reloaded = afir.load %input outs(%load_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed2 = afir.add %reloaded, %reloaded outs(%compute_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>,
                            #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %computed2 outs(%workspace) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    %back = afir.load %workspace outs(%load_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    afir.store %back outs(%output) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }

  func.func private @store_nddma_workspace(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %workspace: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %output: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], role = <input>>,
      #afir.tensor_desc<axis = [0], role = <output>>,
      #afir.tensor_desc<axis = [0], role = <output>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %load_tile = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2,
                                                            slot_count = 2, queue_depth = 2,
                                                            size_bytes_expr = <"(2 * s0)">,
                                                            alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %compute_tile = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3,
                                                            slot_count = 2, queue_depth = 2,
                                                            size_bytes_expr = <"(2 * s0)">,
                                                            alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %input outs(%load_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed = afir.add %loaded, %loaded outs(%compute_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>,
                            #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %computed outs(%workspace) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    %back = afir.nddma %workspace outs(%load_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed2 = afir.add %back, %back outs(%compute_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>,
                            #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %computed2 outs(%output) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// SYNC-LABEL: func.func private @store_load_workspace
// 回读链：store 写 workspace 之后 record，load 回读之前 wait。
// SYNC: %[[WS:.*]] = afir.event_record : <from = transfer_out, to = transfer_in>
// SYNC: afir.event_wait %[[WS]] : <from = transfer_out, to = transfer_in>
// SYNC: afir.load
// SYNC: afir.add
// SYNC: afir.store

// MULTI-LABEL: func.func private @multi_writer_readback
// 同一 workspace 根上两个 store（多 stage 各写一次的形态）后一次回读：GM 写者全在
// MTE3，末写者的 record 按程序序覆盖前一个 store——只有一条事件对，且 record 紧跟
// 第二个 store。两条 NEXT 断言把归并钉死：首个 store 之后直接是 reloaded
// load（store1 与它之间没有 record），末个 store 之后直接是唯一的 record；
// 逐写者建对的旧形态会在 store1 后插一条 record，第一条 NEXT 即失败。
// MULTI: afir.store
// MULTI-NEXT: afir.load
// MULTI: afir.add
// MULTI: afir.store
// MULTI-NEXT: %[[WS:.*]] = afir.event_record : <from = transfer_out, to = transfer_in>
// MULTI: afir.event_wait %[[WS]] : <from = transfer_out, to = transfer_in>
// MULTI: afir.load

// NDDMA-LABEL: func.func private @store_nddma_workspace
// 与 SYNC 同构，只是回读者换成 nddma：store 之后 record、nddma 之前 wait，一对都不能少。
// NDDMA: afir.store
// NDDMA-NEXT: %[[WS:.*]] = afir.event_record : <from = transfer_out, to = transfer_in>
// NDDMA-NEXT: afir.event_wait %[[WS]] : <from = transfer_out, to = transfer_in>
// NDDMA: afir.nddma
// NDDMA: afir.add
// NDDMA: afir.store
