// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// AR 单 A 的物理布局证明（设计文档 `2026-10-13-reduce-micro-refactor-design.md` 功能需求 3）。
//
// 这是一组差分用例：三份 IR 只差一个 stride，第一份必须通过、后两份必须被拒。只断言
// 「能不能过」而不是报错措辞——差异唯一，拒绝只可能来自那一处。
//
// 为什么需要它：`materializePhysicalReduceShape` 自己构造
// `innerFootprint = align(N_Rlast) × Π N_Rother` 喂给高阶 API，而 tile 的真实 stride 来自
// 调度器。两者不一致时 API 按错的行距走位，算的是补白里的垃圾，而此前没有任何检查会报：
// 这几条判据以前只写在注释里。
//
// R 取常量 17、对齐宽度 8，于是 `align(17) = 24`——合法行距与裸长度差得开，差分才看得见。
// 两根轴都是 `Original`，所以 nominal 恒等于 actual，不需要 tiling 参数和 axis_binding；
// `loop_axis` 取在归约轴之外，所以不跨轮累加，也就不需要真实的 `scf.for`。

// 两条 RUN 跑同一条命令、查两件事：合法那份下降出来了，另两份各被拒一次。整体退出码非零
// （后两份报错），所以两条都要 `not`。
// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 --split-input-file %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LEGAL
// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 --split-input-file %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REJECT

// 第一份与第六份合法：A 的 stride 恰是对齐后的 R footprint（24），落点 A 紧凑写、R 不前进；
// 第六份是自洽的退化保留轴（全归约形态）。
// LEGAL-COUNT-2: ascendc.reduce_sum_2d_l2

// 后四份各被拒一次。能力查询在 SelectImplementation 处就拦下，报的是通用那句。
// REJECT-COUNT-4: error: 'afir.sum' op has no supported Ascend implementation

module {
func.func private @ar_layout_legal(
  %src: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"384">, life_time_axis_id = -1>},
  %dst: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"32">, life_time_axis_id = -1>},
  %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"256">, life_time_axis_id = -1>},
  %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "a", axis_type = Original, size = <"4">, align = <"1">>,
 <id = 1, name = "r", axis_type = Original, size = <"17">, align = <"1">>
 ], type = Compute, size_var = [<"unused">]>
} {
 %result = afir.sum %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"17">], strides = [<"24">, <"1">], vectorized_axis = [0, 1], vectorized_strides = [<"24">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">], strides = [<"1">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"1">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">], strides = [<"1">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"1">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [0, 1], loop_axis = 0>
 } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf32>
 return
}
}

// -----

// 源侧 A 的 stride 取未对齐的裸长度 17。第二行起就偏进了前一行的补白，而高阶 API 仍按
// 对齐后的 24 走行距。唯一改动：`vectorized_strides` 第一项 24 → 17。
module {
func.func private @ar_layout_unaligned_source_pitch(
  %src: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"384">, life_time_axis_id = -1>},
  %dst: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"32">, life_time_axis_id = -1>},
  %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"256">, life_time_axis_id = -1>},
  %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "a", axis_type = Original, size = <"4">, align = <"1">>,
 <id = 1, name = "r", axis_type = Original, size = <"17">, align = <"1">>
 ], type = Compute, size_var = [<"unused">]>
} {
 %result = afir.sum %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"17">], strides = [<"24">, <"1">], vectorized_axis = [0, 1], vectorized_strides = [<"17">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">], strides = [<"1">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"1">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">], strides = [<"1">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"1">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [0, 1], loop_axis = 0>
 } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf32>
 return
}
}

// -----

// 落点侧 A 的 stride 是 2 而不是 1。高阶 API 把结果连续写在开头，消费侧却按跨步 2 读，
// 于是一半结果读到的是没写过的槽位。只验证源侧是不够的——这条正是上一轮检视里
// 「destination 的 nominal stride 不会自动因尾块压紧」那个发现的门禁化。
// 唯一改动：destination / result 的 `vectorized_strides` 第一项 1 → 2。
module {
func.func private @ar_layout_strided_destination(
  %src: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"384">, life_time_axis_id = -1>},
  %dst: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"64">, life_time_axis_id = -1>},
  %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"256">, life_time_axis_id = -1>},
  %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "a", axis_type = Original, size = <"4">, align = <"1">>,
 <id = 1, name = "r", axis_type = Original, size = <"17">, align = <"1">>
 ], type = Compute, size_var = [<"unused">]>
} {
 %result = afir.sum %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"17">], strides = [<"24">, <"1">], vectorized_axis = [0, 1], vectorized_strides = [<"24">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">], strides = [<"2">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"2">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">], strides = [<"2">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"2">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [0, 1], loop_axis = 0>
 } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf32>
 return
}
}

// -----

// graph 与 descriptor 的 extent 不同源（2026-10-13 检视的 P1 回归）。唯一改动：graph 里
// R 的 size 由 17 变成 33，descriptor 的 `repeats` 仍是 17、A stride 仍是 24。
//
// 按 descriptor 的 `repeats` 证明会得到 `24 == align(17)` 成立而放行；但高阶 API 的行距
// 走的是 graph 的 `align(33) = 40`，连 384 字节的源缓冲都装不下生成的四行。判据因此必须从
// `materializeScheduledAxisExtents` 实际用的那一份（`AxisAttr::getSize()`）读 extent。
module {
func.func private @ar_layout_graph_extent_mismatch(
  %src: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"384">, life_time_axis_id = -1>},
  %dst: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"32">, life_time_axis_id = -1>},
  %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"256">, life_time_axis_id = -1>},
  %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "a", axis_type = Original, size = <"4">, align = <"1">>,
 <id = 1, name = "r", axis_type = Original, size = <"33">, align = <"1">>
 ], type = Compute, size_var = [<"unused">]>
} {
 %result = afir.sum %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"17">], strides = [<"24">, <"1">], vectorized_axis = [0, 1], vectorized_strides = [<"24">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">], strides = [<"1">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"1">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">], strides = [<"1">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"1">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [0, 1], loop_axis = 0>
 } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf32>
 return
}
}

// -----

// 退化判定也必须按 graph 的 extent（同一轮检视的第二个反例）。唯一改动：destination /
// result 的 A `repeat` 由 4 变成 1，stride 由 1 变成 2；graph 与 input 仍是 A=4。
//
// 按 destination 的 `repeat` 判退化会以为「只有一行、没有行间距要验证」而跳过检查；而 API
// 按 graph 的 A=4 连续写四个结果，与 stride 2 完全对不上。
module {
func.func private @ar_layout_destination_degenerate_mismatch(
  %src: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"384">, life_time_axis_id = -1>},
  %dst: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"32">, life_time_axis_id = -1>},
  %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"256">, life_time_axis_id = -1>},
  %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "a", axis_type = Original, size = <"4">, align = <"1">>,
 <id = 1, name = "r", axis_type = Original, size = <"17">, align = <"1">>
 ], type = Compute, size_var = [<"unused">]>
} {
 %result = afir.sum %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"17">], strides = [<"24">, <"1">], vectorized_axis = [0, 1], vectorized_strides = [<"24">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"1">, <"1">], strides = [<"2">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"2">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"1">, <"1">], strides = [<"2">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"2">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [0, 1], loop_axis = 0>
 } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf32>
 return
}
}

// -----

// 正向：自洽的退化保留轴。graph 里 A 的 size 就是 1，两侧 stride 都是 0——这是全归约的
// 合法形态（对应 `ReducePlanTest.AllReduceKeepsTheDegenerateAxisAsTheKeptSide`）。只有一行，
// 不存在行间距可验证，所以跳过 stride 约束是对的，不能因为 stride 不是 1 就拒。
module {
func.func private @ar_layout_degenerate_kept_axis(
  %src: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"384">, life_time_axis_id = -1>},
  %dst: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"32">, life_time_axis_id = -1>},
  %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"256">, life_time_axis_id = -1>},
  %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "a", axis_type = Original, size = <"1">, align = <"1">>,
 <id = 1, name = "r", axis_type = Original, size = <"17">, align = <"1">>
 ], type = Compute, size_var = [<"unused">]>
} {
 %result = afir.sum %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"1">, <"17">], strides = [<"0">, <"1">], vectorized_axis = [0, 1], vectorized_strides = [<"0">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"1">, <"1">], strides = [<"0">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"0">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"1">, <"1">], strides = [<"0">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"0">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [0, 1], loop_axis = 0>
 } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf32>
 return
}
}
