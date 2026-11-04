// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// **真实 dump 的两段 `afir.broadcast`：整条链路端到端落下来。**
//
// 这个文件此前是反例（「多轴那一档必须报 unsupported」）。descriptor 一个字没改——改的是
// 判据：`TransferPlan` 的单轴假设被 `collapseVectorizedAxes` 取代之后，这两根向量化轴
// 合成两段，正好是形态 A `[brc K][normal Z]`。
//
// 源在 zt 上 repeat 1 / stride 0（广播段），在 z1 上 repeat 8 / stride 1（普通段）；
// 目的地两段的 stride 是 8 与 1。两段不合并不是特判，是同一条 stride 连续性判据的结果：
// 源这一侧 `0 == 1 * 8` 不成立。
//
// 这一份钉的是**三个算子在同一张图上各自读对了同一批 stride**：
//
//   - **Load**：源在切分轴上 stride 为 0 → GM 偏移恒为 0（每趟读同一份 8 个元素），
//     搬运长度是 8（丢掉广播段之后剩下那一段的 extent），**不是**切分轴的轴长；
//   - **Broadcast**：发 `Broadcast(dst, src, 1, Z, 0, K, Z, 0, tmp)`，落 `BroadcastFirstDim`
//     （只有 `Copy`）。legacy 在同一个形状上走三轴形态，`Z > 8` 时落 `BroadcastMiddleDim`
//     的 `Or`。**第三个实参恒为 `1`** 是那条不变量在编译期唯一钉得住的形式；
//   - **Store**：目的地是两段合成的一段 → 每趟前进 `zt_size × 8`、搬 `尾块长度 × 8`，
//     **不是**轴长。
//
// 三处此前都取不到：Load 与 Store 走的 `materializeScheduledTileIndices` 当时按
// 「`vectorized_strides.front()` 是 0 还是 1」分流、按**轴长**算地址与元素数——Load 命中
// 0 那一档（那一档还要求 GM stride 全为 0，而这里的 GM 输入是连续 8 个 float），
// Store 的 `front()` 是 8、既非 0 也非 1，两边都报失败。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=broadcast_two_segment %t.mlir \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT

#gm_in = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"8">], strides = [#afir.expr<"1">], role = <input>>
#gm_out = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1024">], strides = [#afir.expr<"1">], role = <output>>
// 两根向量化轴：zt 是广播轴（repeat 1 / stride 0），z1 不是（repeat 8 / stride 1）。
#src_tile = #afir.tensor_desc<axis = [3, 4, 2, 5],
                              repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"8">],
                              strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">, #afir.expr<"1">],
                              vectorized_axis = [2, 5],
                              vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst_tile = #afir.tensor_desc<axis = [3, 4, 2, 5],
                              repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                         #afir.expr<"zt_size">, #afir.expr<"8">],
                              strides = [#afir.expr<"(8 * zTb_size * zt_size)">, #afir.expr<"(8 * zt_size)">,
                                         #afir.expr<"8">, #afir.expr<"1">],
                              vectorized_axis = [2, 5],
                              vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input: memref<8xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                             afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"32">,
                                                                  life_time_axis_id = -1>},
      %output: memref<1024xf32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>,
                                 afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4096">,
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
               split_pair_other_id = 3>,
              <id = 5, name = "z1", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 5>],
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
                                                                life_time_axis_id = -1>} : memref<1x1x1x8xf32>
    %loaded = afir.load %input outs(%src) {
      afir.destination_descs = [#src_tile],
      afir.operand_descs = [#gm_in],
      afir.result_descs = [#src_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2, 5], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>]
    } : memref<8xf32>, memref<1x1x1x8xf32> -> memref<1x1x1x8xf32>
    %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?x8xf32>
    %tmp = memref.alloc(%tmp_bytes) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                                          size_from_resource = true,
                                                                          alignment_bytes = 32,
                                                                          life_time_axis_id = -1>} : memref<?xui8>
    %broadcast = afir.broadcast %loaded outs(%dst) tmp(%tmp) {
      afir.destination_descs = [#dst_tile],
      afir.operand_descs = [#src_tile],
      afir.result_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2, 5], loop_axis = 4>
    } : memref<1x1x1x8xf32>, memref<?x?x?x8xf32>, memref<?xui8> -> memref<?x?x?x8xf32>
    afir.store %broadcast outs(%output) {
      afir.destination_descs = [#gm_out],
      afir.operand_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2, 5], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d4)>]
    } : memref<?x?x?x8xf32>, memref<1024xf32>
    return
  }
}

// IR-LABEL: func.func private @stage0_template0
// 只有 tile 循环这一层：两段全部交给两轴 API，广播不再额外建循环。
// IR: scf.for
// IR-NOT: scf.for
// IR: %[[K:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 2, role = actual_size>}

// Load：GM 偏移是常量 0（源不随 tile 前进），搬运长度是 8——**不是**切分轴的轴长，
// 也**不是**广播轴上的 repeat（那是 1）。
// IR: %[[SRC_OFF:.*]] = arith.constant 0 : index
// IR: %[[SRC_LEN:.*]] = arith.constant 8 : index
// IR: %[[SRC_GM:.*]] = ascendc.global_tensor.subindex %{{.*}}[%[[SRC_OFF]]]
// IR: %[[SRC_UB:.*]] = ascendc.local_tensor.subindex
// IR: ascendc.data_copy_pad_extend %[[SRC_UB]], %[[SRC_GM]], %{{.*}}, %[[SRC_LEN]], %

// Broadcast：形态 A，第三个实参是 `1`，趟数是带尾块修正的 K，`src_z` / `dst_z` 都是 0。
// IR: %[[BC_Z:.*]] = arith.constant 8 : index
// IR: %[[BC_ZERO:.*]] = arith.constant 0 : index
// IR: %[[BC_ONE:.*]] = arith.constant 1 : index
// IR: ascendc.broadcast_dim3 %{{.*}}, %{{.*}}, %[[BC_ONE]], %[[BC_Z]], %[[BC_ZERO]], %[[K]], %[[BC_Z]], %[[BC_ZERO]], %

// Store：GM 偏移由分核与循环两项求和，搬运长度为 `actual(zt) × 8`。
// IR: %[[BLOCK_STEP0:.*]] = index.mul %{{.*}}, %{{.*}}
// IR: %[[BLOCK_STEP:.*]] = index.mul %[[BLOCK_STEP0]], %{{.*}}
// IR: %[[BLOCK_OFF:.*]] = arith.muli %{{.*}}, %[[BLOCK_STEP]] : index
// IR: %[[LOOP_OFF:.*]] = arith.muli %{{.*}}, %{{.*}} : index
// IR: %[[GM_OFF:.*]] = arith.addi %[[BLOCK_OFF]], %[[LOOP_OFF]] : index
// IR: %[[LEN:.*]] = index.mul %{{.*}}, %[[K]]
// IR: %[[DST_UB:.*]] = ascendc.local_tensor.subindex
// IR: %[[DST_GM:.*]] = ascendc.global_tensor.subindex %{{.*}}[%[[GM_OFF]]]
// IR: ascendc.data_copy_pad_extend %[[DST_GM]], %[[DST_UB]], %{{.*}}, %[[LEN]], %

// kernel 文本侧：一条 `Broadcast`，第三个实参是 `1`。
// CPP: DataCopyPadExtend<float>(
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(v{{[0-9]+}}), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}});
// CPP: DataCopyPadExtend<float>(

// 选中的是分段那一档，不是 `Duplicate` 那一档——源不是一个值，选错会静默把源的第 0 个
// 元素铺满整块目的地。
// SELECT: ascend.implementation = {id = "broadcast.tensor.segmented", kind = "ascendc_api"}
