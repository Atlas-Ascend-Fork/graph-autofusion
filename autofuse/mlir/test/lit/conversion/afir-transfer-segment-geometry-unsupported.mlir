// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 跨段 load/store 的 **UB 侧段几何防御判据**：行距小于行元素数（行重叠）与行距
// 不整除（推进漂移）。
//
// **推进模型（探针实证）**：`DataCopyPadExtend` 的 UB 侧每块推进是
// `AlignUp_g(行长) + g * trunc((行距 - 行长) / g)`（g = 32B/元素；发射把缝隙
// `行距 - 行长` 物化为 SubI，包装 `dst_stride / align_num` 整除截断，`datacopy.h`）。
// CPU 仿真与 NPU 同源 dav_c220 头，单文件探针钉死三组数：缝隙 0/行长 37 推进 40、
// 缝隙 11 推进 48、缝隙 259 推进 296。推进恰等于名义行距 ⟺ 行距 ≡ 0 (mod g)。
//
// **分方向**：
//  - **load（UB 写侧）**：布局由这块 DMA 落出来、消费者按 desc 行距读。行距 ≢ 0
//    (mod g) 时写出的布局与 desc 自相矛盾——上线证据：pitch 296/行长 37（缝隙 259）
//    的转置读 NPU bit 级通过，296 = AlignUp(37) + 256。
//  - **store（UB 读侧）**：物理行距由生产者决定（NDDMA 按元素 stride 精确写），
//    本 op 的 desc 看不见。行距 == 行长（缝隙 0）时 stride 参数精确无损，放行——
//    调度器会为**未选中的候选**发射这种形态（reg_base plan1 的 store，tiling_key
//    分发后是死代码但必须编过，正例由该 e2e 钉住）。其余形态拒行距 ≢ 0 (mod g)
//    ——上线证据：pitch 48/行长 37（缝隙 11）通过，48 = 40 + 8。
//
// **只判 UB 侧，不判 GM 侧**：GM 侧负缝隙是转置读写模板的**合法上线形态**（行维
// GM 步长 1、行长 37，`kernel.cpp` 里 `c1_idx - c37_idx` 那条），NPU bit 级验证过；
// 设计早期「负跨度在合轴层就拒」的理论只对 UB 侧成立。
//
// **判据方向：可证违反才拒**（符号形态答不了就放行），与合轴的保守方向一致——合法
// 调度产物里行距大量是符号形态（`16 * Ceiling(...)`），倍数性答不了是常态。这与
// `afir-remove-pad-unsupported.mlir` 的整除档方向**相反**：那侧行距恒为对齐形态、
// 「非可证倍数即拒」；两侧 `isProvablyMultipleOf` / `isProvenNotMultipleOf` 的注释互指。
//
// **判据住在 `blockDmaSegmentGeometryIsSound`**（`TransferPlan.cpp`），DataMovement
// 的能力层与 pattern 共用同一份；作用在喂给 DMA 的那一对维上（行维 + 最内维），更
// 外层维只生成循环偏移、不进 DMA 参数。
//
// 正例对照是 `afir-transfer-multi-segment-unsupported.mlir`：同一模板、GM 行距 8 ==
// 行长、UB 行距 16（8 的倍数），两维照常下降。三个分片各违反一条、互不混叠：
//   - `@neg_local_span_load`：UB 行距 8 < 行长 12（GM 行距 12 合法、8 是 8 的倍数）
//     ——只踩行重叠。
//   - `@misaligned_pitch_load`：UB 行距 9 > 行长 8、两侧行距都不小于行长——只踩
//     load 侧整除。
//   - `@misaligned_pitch_store`：同一几何形态换成 store（UB 行距 9 > 行长 8、行距
//     ≠ 行长）——钉住 store 侧没有「行距 == 行长」豁免时的整除拒绝。

// RUN: not af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation --split-input-file %s 2>&1 \
// RUN:   | FileCheck %s

#gm_twelve = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"12">],
                               strides = [#afir.expr<"12">, #afir.expr<"1">], role = <input>>
// UB 侧行距 8：是 8 的倍数、GM 侧行距 12 == 行长，只有 UB 侧行距小于行长（8 < 12）。
#neg_local_tile = #afir.tensor_desc<axis = [3, 4, 2, 1],
                          repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                     #afir.expr<"zt_size">, #afir.expr<"12">],
                          strides = [#afir.expr<"(12 * zTb_size * zt_size)">, #afir.expr<"(12 * zt_size)">,
                                     #afir.expr<"12">, #afir.expr<"1">],
                          vectorized_axis = [2, 1],
                          vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">],
                          role = <compute>>

func.func private @neg_local_span_load(
    %input: memref<128x12xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                                afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"6144">,
                                                                     life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm_twelve],
  afir.asc_graph = #afir.asc_graph<
    tiling_key = 0,
    axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"128">, split_pair_other_id = 0>,
            <id = 1, name = "w", axis_type = Original, size = #afir.expr<"12">, split_pair_other_id = 1>,
            <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">, from = [0],
             split_pair_other_id = 5>,
            <id = 3, name = "zTB", axis_type = BlockOuter,
             size = #afir.expr<"Ceiling((Ceiling((128 / (zt_size))) / (zTb_size)))">, from = [5],
             split_pair_other_id = 4>,
            <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">, from = [5],
             split_pair_other_id = 3>,
            <id = 5, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((128 / (zt_size)))">,
             from = [0], split_pair_other_id = 2>],
    type = Compute,
    size_var = [#afir.expr<"zt_size">, #afir.expr<"zTb_size">]>,
  afir.function_role = #afir.function_role<kernel_body>
} {
  %tile_size = afir.tiling_param "zt_size" : index
  %block = afir.tiling_param "zTb_size" : index
  %outer = afir.symbolic_eval %block, %tile_size {expr = #afir.expr<"(128 / (__arg0 * __arg1))">}
      : (index, index) -> index
  %dst = memref.alloc(%outer, %block, %tile_size) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?x12xf32>
  %loaded = afir.load %input outs(%dst) {
    afir.destination_descs = [#neg_local_tile],
    afir.operand_descs = [#gm_twelve],
    afir.result_descs = [#neg_local_tile],
    afir.sched = #afir.sched<axis = [3, 4, 2, 1], loop_axis = 4>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>]
  } : memref<128x12xf32>, memref<?x?x?x12xf32> -> memref<?x?x?x12xf32>
  return
}

// -----

#gm_in = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"8">],
                           strides = [#afir.expr<"8">, #afir.expr<"1">], role = <input>>
// UB 侧行距 9：大于行长、两侧行距都不小于行长，只有「9 不是 8 的倍数」这一条被踩。
#misaligned_tile = #afir.tensor_desc<axis = [3, 4, 2, 1],
                          repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                     #afir.expr<"zt_size">, #afir.expr<"8">],
                          strides = [#afir.expr<"(8 * zTb_size * zt_size)">, #afir.expr<"(8 * zt_size)">,
                                     #afir.expr<"8">, #afir.expr<"1">],
                          vectorized_axis = [2, 1],
                          vectorized_strides = [#afir.expr<"9">, #afir.expr<"1">],
                          role = <compute>>

func.func private @misaligned_pitch_load(
    %input: memref<128x8xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                               afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4096">,
                                                                    life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm_in],
  afir.asc_graph = #afir.asc_graph<
    tiling_key = 0,
    axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"128">, split_pair_other_id = 0>,
            <id = 1, name = "w", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 1>,
            <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">, from = [0],
             split_pair_other_id = 5>,
            <id = 3, name = "zTB", axis_type = BlockOuter,
             size = #afir.expr<"Ceiling((Ceiling((128 / (zt_size))) / (zTb_size)))">, from = [5],
             split_pair_other_id = 4>,
            <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">, from = [5],
             split_pair_other_id = 3>,
            <id = 5, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((128 / (zt_size)))">,
             from = [0], split_pair_other_id = 2>],
    type = Compute,
    size_var = [#afir.expr<"zt_size">, #afir.expr<"zTb_size">]>,
  afir.function_role = #afir.function_role<kernel_body>
} {
  %tile_size = afir.tiling_param "zt_size" : index
  %block = afir.tiling_param "zTb_size" : index
  %outer = afir.symbolic_eval %block, %tile_size {expr = #afir.expr<"(128 / (__arg0 * __arg1))">}
      : (index, index) -> index
  %dst = memref.alloc(%outer, %block, %tile_size) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?x8xf32>
  %loaded = afir.load %input outs(%dst) {
    afir.destination_descs = [#misaligned_tile],
    afir.operand_descs = [#gm_in],
    afir.result_descs = [#misaligned_tile],
    afir.sched = #afir.sched<axis = [3, 4, 2, 1], loop_axis = 4>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>]
  } : memref<128x8xf32>, memref<?x?x?x8xf32> -> memref<?x?x?x8xf32>
  return
}

// -----

#gm_out = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"8">],
                            strides = [#afir.expr<"8">, #afir.expr<"1">], role = <output>>
// store 侧同一几何形态：UB 行距 9 > 行长 8、行距 ≠ 行长（无「缝隙 0」豁免）。
#misaligned_src_tile = #afir.tensor_desc<axis = [3, 4, 2, 1],
                          repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                     #afir.expr<"zt_size">, #afir.expr<"8">],
                          strides = [#afir.expr<"(8 * zTb_size * zt_size)">, #afir.expr<"(8 * zt_size)">,
                                     #afir.expr<"8">, #afir.expr<"1">],
                          vectorized_axis = [2, 1],
                          vectorized_strides = [#afir.expr<"9">, #afir.expr<"1">],
                          role = <compute>>

func.func private @misaligned_pitch_store(
    %output: memref<128x8xf32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>,
                                afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4096">,
                                                                     life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm_out],
  afir.asc_graph = #afir.asc_graph<
    tiling_key = 0,
    axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"128">, split_pair_other_id = 0>,
            <id = 1, name = "w", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 1>,
            <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">, from = [0],
             split_pair_other_id = 5>,
            <id = 3, name = "zTB", axis_type = BlockOuter,
             size = #afir.expr<"Ceiling((Ceiling((128 / (zt_size))) / (zTb_size)))">, from = [5],
             split_pair_other_id = 4>,
            <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">, from = [5],
             split_pair_other_id = 3>,
            <id = 5, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((128 / (zt_size)))">,
             from = [0], split_pair_other_id = 2>],
    type = Compute,
    size_var = [#afir.expr<"zt_size">, #afir.expr<"zTb_size">]>,
  afir.function_role = #afir.function_role<kernel_body>
} {
  %tile_size = afir.tiling_param "zt_size" : index
  %block = afir.tiling_param "zTb_size" : index
  %outer = afir.symbolic_eval %block, %tile_size {expr = #afir.expr<"(128 / (__arg0 * __arg1))">}
      : (index, index) -> index
  %src = memref.alloc(%outer, %block, %tile_size) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?x8xf32>
  afir.store %src outs(%output) {
    afir.destination_descs = [#gm_out],
    afir.operand_descs = [#misaligned_src_tile],
    afir.sched = #afir.sched<axis = [3, 4, 2, 1], loop_axis = 4>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>]
  } : memref<?x?x?x8xf32>, memref<128x8xf32>
  return
}

// 三个分片各拒一条；`--split-input-file` 会跑完全部分片再退出，三条消息钉住三格。
// CHECK: 'afir.load' op has no supported Ascend implementation
// CHECK: 'afir.load' op has no supported Ascend implementation
// CHECK: 'afir.store' op has no supported Ascend implementation
