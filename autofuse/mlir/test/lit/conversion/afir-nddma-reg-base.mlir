// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.nddma` → reg 侧的 `DataCopyNddma<T, dim>`：搬入与广播一条指令。
//
// **三个 func 的 descriptor 是故意挑的，第三个才是本档的设计内容**：
//
//   - `@discontinuous_tile` 与 `afir-transfer-multi-segment-unsupported.mlir` 里那一格**逐字
//     相同**，只把 `afir.load` 换成 `afir.nddma`。那一格是「合轴后两段、`DataCopyPadExtend`
//     表达不了、能力层报不支持」的反例；同一批 descriptor 在这里是**正例**，因为多维恰好是
//     ND-DMA 存在的理由。两个文件一起读才说得完整：不是「多段不支持」，是「多段要换一条路发」。
//   - `@broadcast_tile` 在那之上把 GM 内轴的 stride 置 0（源那一维 repeat 为 1），也就是
//     `NddmaTemplate` 把 Broadcast 融进 Load 之后的样子。`input_stride` 那一位要发 0。
//     片上两轴仍然不连续（行距 16、行长 8），所以合轴照样给两段，分维不必动手。
//   - `@dense_broadcast_tile` 把片上行距改成 8 —— **片上两轴这就连续了，合轴把它们并成一段**，
//     而 GM 侧内轴 stride 是 0、外轴是 1，并不连续。这一格是唯一测得出「分维要问两侧」的：
//     不在 GM 断处切开的话，发出来的是**一维** `output_dims = {zt × 8}`、`input_stride = {1}`，
//     于是把 GM 上连续的 `zt × 8` 个元素搬进来，而正确结果是同一个元素重复 8 次。
//     **编得过、跑得完、结果错。**
//
// **这一层钉「分成几维」，不钉每一维的数值。** 三组尺寸在这个阶段还是 SSA 值
// （`arith.constant` 加 tiling 参数），而同一个数在一个函数里有好几份、名字带各自的后缀
// （`%c8` / `%c8_9`），按名字钉等于钉住编号分配；`af-opt` 又没注册 `--cse`，合不掉。
// 发出来的 C++ 文本在这里也拿不到：这几个 func 是最小用例，没有 workspace / TilingData 形参，
// 走完整 pipeline 会先撞上「canonical AscendC kernel ABI」那道检查（实测），而
// `ascir-translate` 要的是走完 pipeline 之后的 IR。
//
// 所以分工是：
//
//   - **维数**由本文件钉。它就是本档的设计内容——`@dense_broadcast_tile` 那一格合轴只给
//     一段，发出来必须是**两维**；只有一维就说明 GM 侧那一刀没切，而那一格编得过、跑得完、
//     结果错。
//   - **每一维的数值**由 `test/e2e/cpu/cases/load_broadcast_nddma_store` 钉，而且是拿**数值**
//     钉的：逐元素对着 `input0[inputIndex]` 比。`input_stride` 抄成 UB 的那一位、或者该发
//     非 0 时发了 0，那里当场红。
//   - **发出来的 C++ 长什么样**由 `test/lit/target/ascendc-data-copy-nddma-reg-base.mlir` 钉。
//
// 三组尺寸各自取自哪里：
//
//   `output_dims`   维的元素数（`materializeSegmentExtent`）
//   `output_stride` 维在片上张量上的步长（维内由内往外累乘段的 stride）
//   `input_stride`  维在 GM 上的步长（`afir::globalAxisUnitStride`，合轴**给不了**）

// RUN: af-opt --target=Ascend950DT_950x --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --target=Ascend950DT_950x --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc %s \
// RUN:   | FileCheck %s --check-prefix=IR

#gm_in = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"8">],
                           strides = [#afir.expr<"8">, #afir.expr<"1">], role = <input>>
// 两根向量化轴，外段 stride 16 而内段 extent 8 —— `stride[外] == stride[内] × extent[内]`
// 不成立，合轴封段，这块 tile 剩两段。UB 侧行距 16、GM 侧行距 8，正是 ND-DMA 要表达的那种
// 「两边 pitch 不同」。
#tile = #afir.tensor_desc<axis = [3, 4, 2, 1],
                          repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                     #afir.expr<"zt_size">, #afir.expr<"8">],
                          strides = [#afir.expr<"(16 * zTb_size * zt_size)">, #afir.expr<"(16 * zt_size)">,
                                     #afir.expr<"16">, #afir.expr<"1">],
                          vectorized_axis = [2, 1],
                          vectorized_strides = [#afir.expr<"16">, #afir.expr<"1">],
                          role = <compute>>

// 广播形态：GM 侧内轴 repeat 为 1、stride 为 0（源那一维只有一个元素），落点照常前进。
#gm_brc = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"1">],
                            strides = [#afir.expr<"1">, #afir.expr<"0">], role = <input>>

// 片上紧凑：行距 8 就是行长 8，`stride(外) == stride(内) × extent(内)` 成立，合轴把两根
// 向量化轴**并成一段**。GM 侧外轴 1、内轴 0，`1 != 0 × 8` 不成立——分维要在这里切开。
#dense_tile = #afir.tensor_desc<axis = [3, 4, 2, 1],
                          repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                     #afir.expr<"zt_size">, #afir.expr<"8">],
                          strides = [#afir.expr<"(8 * zTb_size * zt_size)">, #afir.expr<"(8 * zt_size)">,
                                     #afir.expr<"8">, #afir.expr<"1">],
                          vectorized_axis = [2, 1],
                          vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">],
                          role = <compute>>

func.func private @discontinuous_tile(
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
  %loaded = afir.nddma %input outs(%dst) {
    afir.destination_descs = [#tile],
    afir.operand_descs = [#gm_in],
    afir.result_descs = [#tile],
    afir.sched = #afir.sched<axis = [3, 4, 2, 1], loop_axis = 4>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>]
  } : memref<128x8xf32>, memref<?x?x?x8xf32> -> memref<?x?x?x8xf32>
  return
}

func.func private @broadcast_tile(
    %input: memref<128x1xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                               afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"512">,
                                                                    life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm_brc],
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
  %loaded = afir.nddma %input outs(%dst) {
    afir.destination_descs = [#tile],
    afir.operand_descs = [#gm_brc],
    afir.result_descs = [#tile],
    afir.sched = #afir.sched<axis = [3, 4, 2, 1], loop_axis = 4>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>]
  } : memref<128x1xf32>, memref<?x?x?x8xf32> -> memref<?x?x?x8xf32>
  return
}

func.func private @dense_broadcast_tile(
    %input: memref<128x1xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                               afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"512">,
                                                                    life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm_brc],
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
  %loaded = afir.nddma %input outs(%dst) {
    afir.destination_descs = [#dense_tile],
    afir.operand_descs = [#gm_brc],
    afir.result_descs = [#dense_tile],
    afir.sched = #afir.sched<axis = [3, 4, 2, 1], loop_axis = 4>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>]
  } : memref<128x1xf32>, memref<?x?x?x8xf32> -> memref<?x?x?x8xf32>
  return
}

// 三格都选中 reg 侧那一档，而且是 `nddma.*` 而不是 `load.*`——这个 op 上只有这一个
// implementation。
// SELECT-LABEL: func.func private @discontinuous_tile
// SELECT: ascend.implementation = {id = "nddma.gm_ub_reg_base", kind = "ascendc_api"}
// SELECT-LABEL: func.func private @broadcast_tile
// SELECT: ascend.implementation = {id = "nddma.gm_ub_reg_base", kind = "ascendc_api"}
// SELECT-LABEL: func.func private @dense_broadcast_tile
// SELECT: ascend.implementation = {id = "nddma.gm_ub_reg_base", kind = "ascendc_api"}

// 发出来的是一条 `data_copy_nddma_reg_base`，三组各两项。**不发 `data_copy_pad_extend`**：
// 那是单段那条路，两条路不该同时命中。
//
// `operandSegmentSizes` 的后三位就是维数（`1, 1, dim, dim, dim`），这是本文件真正钉的那个数。
// IR-LABEL: func.func private @discontinuous_tile
// IR: ascendc.data_copy_nddma_reg_base
// IR-SAME: operandSegmentSizes = array<i32: 1, 1, 2, 2, 2>
// IR-NOT: ascendc.data_copy_pad_extend

// 片上两轴不连续（行距 16、行长 8），合轴本来就给两段，这一格分维不必动手。
// IR-LABEL: func.func private @broadcast_tile
// IR: ascendc.data_copy_nddma_reg_base
// IR-SAME: operandSegmentSizes = array<i32: 1, 1, 2, 2, 2>
// IR-NOT: ascendc.data_copy_pad_extend

// **这一格是分维判据的反例。** 片上紧凑（行距 8 == 行长 8）所以合轴把两根向量化轴并成
// 一段，而 GM 侧 `1 != 0 × 8`。不切的话这里会是 `1, 1, 1, 1, 1`——一维、`output_dims = {zt × 8}`、
// `input_stride = {1}`，把 GM 上连续的 `zt × 8` 个元素搬进来，而正确结果是同一个元素重复 8 次。
// IR-LABEL: func.func private @dense_broadcast_tile
// IR: ascendc.data_copy_nddma_reg_base
// IR-SAME: operandSegmentSizes = array<i32: 1, 1, 2, 2, 2>
// IR-NOT: ascendc.data_copy_pad_extend

// mem 侧选不出：那一侧的供给表上没有 `DataCopyNddma` 这个包装（`autofuse/ascendc/api/` 下
// 没有对应文件），所以这个 op 在那一侧一律报不支持、整张图回落 legacy。
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WRONGSIDE
// WRONGSIDE: no supported
