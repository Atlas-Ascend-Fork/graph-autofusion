// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.pad`（带 host tiling）→ 一条**运行期**二选一：
// `scf.if (S != E)` 里 `AscendC::Pad(dst, src, PadParams{0, S-E, 0}, tmp, tiling)`
// 对 `AscendC::DataCopy(dst, src, KernelUtils::BlkAlign<T>(元素数连乘))`。
//
// 条件、两支的参数算式逐字搬 legacy `PadApiCall::Generate`
// （`codegen/api_call/elewise/pad_api_call.cpp:61-83`）：S = y 倒数第二根向量化轴的
// 对齐行距、E = y 尾轴实际长度；有补位走 `AscendC::Pad`，没补位退化为整块拷。这里
// 移植为 `scf.if`，不做 legacy 那个「编译期可证无补位就省掉 if」的特例——运行期结果
// 相同，少一个分支。
//
// **fixture 是 4D transpose 链 + pad**，不是 2D E2E 那张真图的形态。真图里 pad 的输入
// 来自 remove_pad、remove_pad 的输入来自一个**三段** load（16 粒度散布，`afir_1` dump
// 的 `vectorized_strides = [16*z0t, 16]` 那格）——那个 load 的下降是 FR3（>2 段外层
// 循环）的范围，本文件不带它，用 `afir.transpose` 的输出喂 pad（on-chip 消费者不挑
// DMA 形态）。pad 自己的判据不依赖上游是谁。
//
// 两份 descriptor 的取法照 bridge 逐字：`strides` 是逻辑（GM 系）寻址、
// `vectorized_strides` 是 UB 侧物理拷贝模式，pad 的 dst 只改后者——z 轴物理步长 3 →
// 16（3 个 f16 的行补到 32B 粒度），`strides` 原样不动（`afir_1` dump 的 pad 同形）。
// S=16、E=3，`padNum=13` 是本格的实算值；else 支的元素数连乘含 TileInner 的 bt 轴
// （尾块修正路径）与三根常量轴。
//
// **为什么 IR 里两个分支都在**：条件是运行期值，`scf.if` 两侧都发射——CPP 那一半
// 同时钉 `PadParams` 两语句形态（官方 API 的 `PadParams&`/`PadTiling&` 是非常量引用，
// braced 临时量绑不上，先落局部再传，`pad_api_call.cpp:66` 同款）与 `BlkAlign` 包装。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.noempty.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=pad_ub_ub %t.noempty.mlir \
// RUN:   | FileCheck %s --check-prefix=NOEMPTY
// NOEMPTY-NOT: #include ""

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 --cse %s | FileCheck %s --check-prefix=IR
// kernel 文本与 TilingData 头由 `--afir-aggregate-launch` 直接出（它自己翻译，不再过
// `ascir-translate`；`ascir-translate` 不认 `index` 方言与残留的 afir 属性，实测过不了）。
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=pad_ub_ub %t.mlir \
// RUN:   | FileCheck %s --check-prefix=CPP
// Axis IDs are graph identities, not fixed positions. Renumber TileInner 4 to 14.
// RUN: sed 's/id = 4,/id = 14,/g;s/split_pair_other_id = 4>/split_pair_other_id = 14>/g;s/\[4,/[14,/g;s/, 4,/, 14,/g' %s | af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 --cse | FileCheck %s --check-prefix=IR
// SELECT 用手工链（`afir-broadcast-duplicate.mlir` 同款分工：candidate pipeline 的产物
// 已带实现属性，但这里单独钉「选中了哪一个」更直观）。
// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT
// 两侧共用同一个官方调用（tier D1），950 上选中的是**同一个** implementation——与
// `afir-remove-pad.mlir` 的 950 行同一条判据。
// RUN: af-opt --target=Ascend950DT_950x --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT950
// Source and destination outer gaps must be rejected independently of the logical GM strides.
// Remove the transpose producer in the source-gap probe so only Pad's layout check is exercised.
// RUN: sed '/^  %%result = afir.transpose/,/^  } : memref/d;s/afir.pad %%result/afir.pad %%mid/;s/#afir.expr<"48">/#afir.expr<"49">/g' %s | not af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's/#afir.expr<"2048">/#afir.expr<"2049">/g' %s | not af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 2>&1 | FileCheck %s --check-prefix=REJECT
// REJECT: 'afir.pad' op has no supported Ascend implementation

// --- 4D `[0, 3, 2, 1]` transpose + pad：行 3 元素补到 16 ------------------------
#axes = #afir.asc_graph<
  tiling_key = 0,
  axes = [<id = 0, name = "b", axis_type = Original, size = #afir.expr<"1024">, split_pair_other_id = 0>,
          <id = 1, name = "c", axis_type = Original, size = #afir.expr<"3">, split_pair_other_id = 1>,
          <id = 2, name = "z", axis_type = Original, size = #afir.expr<"16">, split_pair_other_id = 2>,
          <id = 3, name = "w", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 3>,
          <id = 4, name = "bt", axis_type = TileInner, size = #afir.expr<"bt_size">, from = [0],
           split_pair_other_id = 7>,
          <id = 5, name = "bTB", axis_type = BlockOuter,
           size = #afir.expr<"Ceiling((Ceiling((1024 / (bt_size))) / (bTb_size)))">, from = [7],
           split_pair_other_id = 6>,
          <id = 6, name = "bTb", axis_type = BlockInner, size = #afir.expr<"bTb_size">, from = [7],
           split_pair_other_id = 5>,
          <id = 7, name = "bT", axis_type = TileOuter, size = #afir.expr<"Ceiling((1024 / (bt_size)))">,
           from = [0], split_pair_other_id = 4>],
  type = Compute,
  size_var = [#afir.expr<"bt_size">, #afir.expr<"bTb_size">]>

#gm_in = #afir.tensor_desc<axis = [0, 1, 2, 3],
    repeats = [#afir.expr<"1024">, #afir.expr<"3">, #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"384">, #afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">], role = <input>>
#src_tile = #afir.tensor_desc<axis = [5, 6, 4, 1, 2, 3],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"3">, #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"(384 * bTb_size * bt_size)">, #afir.expr<"(384 * bt_size)">, #afir.expr<"384">,
               #afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">],
    vectorized_axis = [4, 1, 2, 3],
    vectorized_strides = [#afir.expr<"384">, #afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">],
    role = <compute>>
// transpose 的输出 = pad 的输入：紧凑（z 轴物理步长 3 == 尾轴 c 的 3 个元素）。
#mid_tile = #afir.tensor_desc<axis = [5, 6, 4, 3, 2, 1],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"8">, #afir.expr<"16">, #afir.expr<"3">],
    strides = [#afir.expr<"(384 * bTb_size * bt_size)">, #afir.expr<"(384 * bt_size)">, #afir.expr<"384">,
               #afir.expr<"16">, #afir.expr<"3">, #afir.expr<"1">],
    vectorized_axis = [4, 3, 2, 1],
    vectorized_strides = [#afir.expr<"384">, #afir.expr<"48">, #afir.expr<"3">, #afir.expr<"1">],
    role = <compute>>
// pad 的输出：只改 `vectorized_strides`——z 轴物理步长 3 → 16（3 个 f16 = 6 B，补到
// 32 B 粒度），`strides` 原样（逻辑寻址不变，bridge 同款）。
#pad_dst_tile = #afir.tensor_desc<axis = [5, 6, 4, 3, 2, 1],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"8">, #afir.expr<"16">, #afir.expr<"3">],
    strides = [#afir.expr<"(384 * bTb_size * bt_size)">, #afir.expr<"(384 * bt_size)">, #afir.expr<"384">,
               #afir.expr<"16">, #afir.expr<"3">, #afir.expr<"1">],
    vectorized_axis = [4, 3, 2, 1],
    vectorized_strides = [#afir.expr<"2048">, #afir.expr<"256">, #afir.expr<"16">, #afir.expr<"1">],
    role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
func.func private @pad_after_transpose(
    %input: memref<1024x3x16x8xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                                     afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"786432">,
                                                                          life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm_in],
  afir.asc_graph = #axes,
  // host tiling 字段：bridge 的 `<nodeName>_tilingData_<tiling_case_id>` 形态；同一模板
  // 里 transpose 与 pad 各占一个字段（`GetApiTilingInfo` 两类都发）。成员名：`PadTiling`
  // 抄 `AscendC::tiling::PadTiling` 的 26 个 param（`PadApiTilingFields`），转置是 18 个
  // 占位 param。
  afir.api_tiling_fields = [{field = "transpose_tilingData_0", type = "ConfusionTransposeTiling",
                             fields = ["param0", "param1", "param2", "param3", "param4", "param5",
                                       "param6", "param7", "param8", "param9", "param10", "param11",
                                       "param12", "param13", "param14", "param15", "param16", "param17"]},
                            {field = "pad_tilingData_0", type = "PadTiling",
                             fields = ["srcHeight", "srcWidth", "srcOriWidth", "widthWithoutLastBlock",
                                       "blocksPerRow", "heightTiling", "heightFractal", "heightFractalTail",
                                       "mainLoopOffset", "tailBlockOffset", "tmpBuffer1BlockNum",
                                       "tmpBuffer1RowNum", "tmpBuffer2Offset", "widthTiling", "widthFractal",
                                       "widthFractalTail", "widthFractalTailAlingned", "brcbTiling",
                                       "brcbFractal", "brcbFractalTail", "maxRepeatTimes",
                                       "brcbTilingRepeatTimes", "brcbTilingRepeatTimesTail",
                                       "brcbFractalTailRepeatTimes", "brcbFractalTailRepeatTimesTail",
                                       "reserved"]}],
  afir.function_role = #afir.function_role<kernel_body>,
  afplan.variant = #afplan.variant<stage = 0, template = 0>
} {
  %tile = afir.tiling_param "bt_size" : index
  %block = afir.tiling_param "bTb_size" : index
  %outer = afir.symbolic_eval %block, %tile {expr = #afir.expr<"(1024 / (__arg0 * __arg1))">}
      : (index, index) -> index
  %src = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?x?x?x3x16x8xf16>
  %mid = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>}
      : memref<?x?x?x8x16x3xf16>
  %paddst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 1, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>}
      : memref<?x?x?x8x16x3xf16>
  %tmp = memref.alloc(%tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?xui8>
  %padtmp = memref.alloc(%tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?xui8>
  %loaded = afir.load %input outs(%src) {
    afir.destination_descs = [#src_tile],
    afir.operand_descs = [#gm_in],
    afir.result_descs = [#src_tile],
    afir.sched = #afir.sched<axis = [5, 6, 4, 1, 2, 3], loop_axis = 6>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d1, d2, d3)>]
  } : memref<1024x3x16x8xf16>, memref<?x?x?x3x16x8xf16> -> memref<?x?x?x3x16x8xf16>
  %result = afir.transpose %loaded outs(%mid) tmp(%tmp) {
    afir.operand_descs = [#src_tile],
    afir.destination_descs = [#mid_tile],
    afir.result_descs = [#mid_tile],
    afir.sched = #afir.sched<axis = [5, 6, 4, 1, 2, 3], loop_axis = 6>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5, d4, d3)>],
    api_tiling_field = "transpose_tilingData_0"
  } : memref<?x?x?x3x16x8xf16>, memref<?x?x?x8x16x3xf16>, memref<?xui8> -> memref<?x?x?x8x16x3xf16>
  // bridge 对 elementwise 节点不发 `indexing_maps`（`isTransferNodeType` 只认
  // Load/Store/Nddma/Transpose），`afir.sched` 照发（`appendScheduleAttrs` 对全部
  // buffer-form 节点生效）。
  %padded = afir.pad %result outs(%paddst) tmp(%padtmp) {
    afir.operand_descs = [#mid_tile],
    afir.destination_descs = [#pad_dst_tile],
    afir.result_descs = [#pad_dst_tile],
    afir.sched = #afir.sched<axis = [5, 6, 4, 3, 2, 1], loop_axis = 6>,
    api_tiling_field = "pad_tilingData_0"
  } : memref<?x?x?x8x16x3xf16>, memref<?x?x?x8x16x3xf16>, memref<?xui8> -> memref<?x?x?x8x16x3xf16>
  return
}
}

// SELECT-LABEL: func.func private @pad_after_transpose
// SELECT: ascend.implementation = {id = "transpose.ub_ub_extend", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "pad.ub_ub_extend", kind = "ascendc_api"}

// Check values and SSA uses, not just the presence of arithmetic. CSE shares
// constants, so the same 16 is both the padded pitch and the z-axis extent.
// IR-LABEL: func.func private @pad_after_transpose
// IR: %[[ACTUAL:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = {{4|14}}, role = actual_size>}
// IR: emitasc.member %{{[^ ]+}} "pad_tilingData_0" :
// IR: %[[PITCH:.*]] = arith.constant 16 : index
// IR: %[[WIDTH:.*]] = arith.constant 3 : index
// IR: %[[PADNUM:.*]] = arith.subi %[[PITCH]], %[[WIDTH]] : index
// IR: %[[COND:.*]] = arith.cmpi ne, %[[PITCH]], %[[WIDTH]] : index
// IR: scf.if %[[COND]]
// IR: ascendc.pad {{.*}}, {{.*}}, %[[PADNUM]]
// IR: %[[W:.*]] = arith.constant 8 : index
// IR: %[[BW:.*]] = arith.muli %[[ACTUAL]], %[[W]] : index
// IR: %[[BWZ:.*]] = arith.muli %[[BW]], %[[PITCH]] : index
// IR: %[[COUNT:.*]] = arith.muli %[[BWZ]], %[[WIDTH]] : index
// IR: ascendc.data_copy_blk_aligned {{.*}}, {{.*}}, %[[COUNT]]
// IR-NOT: afir.pad

// include 块四段（kernel 文本的头几行）：官方头（`lib/pad/pad.h`，不内联文本）在固定块
// 后；`utils.h` 是**前导**（排在全部扩展头之前）；`transpose_base_type.h` 必须先于
// `transpose.h`——这一对**反字典序**（`'.'` 的 ASCII 小于 `'_'`），是 include 层按声明
// 序拓扑排（而不是字典序）的那条判据的实证。
// CPP: #include "lib/pad/pad.h"
// CPP: #include "utils.h"
// CPP: #include "transpose_base_type.h"
// CPP-NEXT: #include "transpose.h"
// kernel 文本侧：先落局部 `PadParams`（左补 0、右补 padNum、补值 0——`PadParams` 的
// 声明序 `{leftPad, rightPad, padValue}`，动态位是**第二个**成员）再发调用；`BlkAlign`
// 的模板实参是源元素类型。`scf.if` 两侧都发射（条件是运行期值）。
// CPP: __aicore__ void plan0_pad_after_transpose(
// CPP: AscendC::PadParams padParams_{{[a-zA-Z0-9_]+}} = {0, static_cast<uint16_t>({{[^)]*}}), 0};
// CPP: AscendC::Pad({{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}}, padParams_{{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}});
// CPP: AscendC::DataCopy({{[a-zA-Z0-9_]+}}, {{[a-zA-Z0-9_]+}}, KernelUtils::BlkAlign<half>(static_cast<uint32_t>({{[^)]*}})));
// TilingData 头：`PadTiling` 的 C++ 定义在官方 `kernel_tiling.h` 里，生成头**只引用**
// （成员声明）+ 把 include 打头，自己再定义一份就是与官方头的重定义冲突。
// CPP: #include "kernel_tiling/kernel_tiling.h"
// CPP: PadTiling pad_tilingData_0;
// CPP-NOT: struct PadTiling

// SELECT950-LABEL: func.func private @pad_after_transpose
// SELECT950: ascend.implementation = {id = "pad.ub_ub_extend", kind = "ascendc_api"}
