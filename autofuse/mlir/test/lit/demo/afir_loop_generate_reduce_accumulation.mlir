// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 调度器 S0 通用模板下的 `Load -> Sum(dim=-1) -> Store`，输入 `f32[16, 32]`。函数体逐字取自
// bridge 的真实产物（`schedule plan 0`），没有手工简化——这个用例钉的正是「真实调度
// 长什么样」，简化过的输入钉不住。
//
// 三件事在这里交汇，缺一个都跑不到最后：
//   1. `%9`（sum 的结果）在 z1T 循环里产生，`afir.store` 在外层 z0Tb 消费——跨作用域累加；
//   2. `afir.load` 的 `loop_axis` 是 z1T，即搬运第一次落在**第二层**循环里，且 tile 有两根
//      向量化轴（z0t、z1t），是二维搬运；
//   3. UB 侧的 stride `(8 * Ceiling((Rational(1 , 8) * z1t_size)))` 带 `Rational` 因子。

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate %s | FileCheck %s
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s --check-prefix=ALLOC
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync \
// RUN:   --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6)>
module @reduce_sum_s0 attributes {afir.cube_type = -1 : i64, afir.enable_group_parallel = false, afir.is_reduce_mem_reuse = false, afir.module_kind = "plan", afir.workspace_tensor_ids = array<i64: 2>, afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>, afplan.stages = #afplan.stages<template_counts = [1]>, ascendc.target_aiv_num = 48 : i64, ascendc.target_ub_size = 196608 : i64} {
  func.func private @reduce_sum_common(%arg0: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}, %arg1: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>, #afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"16">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>, <id = 2, name = "z0T", axis_type = TileOuter, size = <"Ceiling((16 / (z0t_size)))">, align = <"1">, from = [0], split_pair_other_id = 3>, <id = 3, name = "z0t", axis_type = TileInner, size = <"z0t_size">, align = <"1">, from = [0], split_pair_other_id = 2>, <id = 4, name = "z1T", axis_type = TileOuter, size = <"Ceiling((32 / (z1t_size)))">, align = <"1">, from = [1], split_pair_other_id = 5>, <id = 5, name = "z1t", axis_type = TileInner, size = <"z1t_size">, align = <"1">, from = [1], split_pair_other_id = 4>, <id = 6, name = "z0TB", axis_type = BlockOuter, size = <"Ceiling((Ceiling((16 / (z0t_size))) / (z0Tb_size)))">, align = <"1">, from = [2], split_pair_other_id = 7>, <id = 7, name = "z0Tb", axis_type = BlockInner, size = <"z0Tb_size">, align = <"1">, from = [2], split_pair_other_id = 6>], type = Compute, size_var = [<"z0t_size">, <"z1t_size">, <"z0Tb_size">]>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_sum_store_16x32_0_B0Y0R1_S0G0C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>], afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    %0 = afir.tiling_param "b0_size" : index
    %1 = afir.tiling_param "b1_size" : index
    %2 = afir.tiling_param "z1t_size" : index
    %3 = afir.tiling_param "z0t_size" : index
    %4 = afir.tiling_param "z0Tb_size" : index
    %5 = afir.symbolic_eval %4, %3 {expr = #afir.expr<"(16 / (__arg0 * __arg1))">} : (index, index) -> index
    %6 = afir.symbolic_eval %2 {expr = #afir.expr<"(32 / (__arg0))">} : (index) -> index
    %7 = afir.symbolic_eval %4, %3 {expr = #afir.expr<"(16 / (__arg0 * __arg1))">} : (index, index) -> index
    %alloc = memref.alloc(%5, %4, %3, %6, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x?x?xf32, #afir.space<ub>>
    %8 = afir.load %arg0 outs(%alloc) {afir.destination_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"(32 / (z1t_size))">, <"z1t_size">],strides = [<"(32 * z0Tb_size * z0t_size)">, <"(32 * z0t_size)">, <"32">, <"z1t_size">, <"1">],vectorized_axis = [3, 5],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>], afir.result_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"(32 / (z1t_size))">, <"z1t_size">],strides = [<"(32 * z0Tb_size * z0t_size)">, <"(32 * z0t_size)">, <"32">, <"z1t_size">, <"1">],vectorized_axis = [3, 5],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"(32 / (z1t_size))">, #afir.expr<"z1t_size">], indexing_maps = [#map, #map1]} : memref<16x32xf32, #afir.space<gm>>, memref<?x?x?x?x?xf32, #afir.space<ub>> -> memref<?x?x?x?x?xf32, #afir.space<ub>>
    %alloc_0 = memref.alloc(%1) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_from_resource = true, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %alloc_1 = memref.alloc(%0) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_from_resource = true, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %alloc_2 = memref.alloc(%7, %4, %3) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x1x1xf32, #afir.space<ub>>
    %9 = afir.sum %8 outs(%alloc_2) scratch(%alloc_0, %alloc_1) {afir.scratch_roles = ["api_scratch", "round_result"], afir.destination_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"(32 / (z1t_size))">, <"z1t_size">],strides = [<"(32 * z0Tb_size * z0t_size)">, <"(32 * z0t_size)">, <"32">, <"z1t_size">, <"1">],vectorized_axis = [3, 5],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.result_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">, #afir.expr<"1">]} : memref<?x?x?x?x?xf32, #afir.space<ub>>, memref<?x?x?x1x1xf32, #afir.space<ub>>, scratch(memref<?xui8, #afir.space<ub>>, memref<?xui8, #afir.space<ub>>) -> memref<?x?x?x1x1xf32, #afir.space<ub>>
    afir.store %9 outs(%arg1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 7>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">, #afir.expr<"1">], indexing_maps = [#map2, #map3]} : memref<?x?x?x1x1xf32, #afir.space<ub>>, memref<16x32xf32, #afir.space<gm>>
    return
  }
}


// CHECK-LABEL: func.func private @reduce_sum_common
// z0TB 是 block 轴，不生成循环；真正的循环是 z0Tb（轴 7）与 z1T（轴 4）两层。
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[LOADED:[a-zA-Z0-9_]+]] = afir.load
// CHECK: %[[SUM:.*]] = afir.sum %[[LOADED]] outs(%[[DST:[a-zA-Z0-9_]+]]) scratch(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}})
// 被归约的 tile 循环在这里闭合——`afir.store` 落在它**外面**。
// CHECK: } {afir.axis_binding = #afir.axis_binding<axis = 4, role = loop>}
// 消费者用的是 sum 的 **destination**，不是 sum 的结果：`%[[SUM]]` 只在循环体内有定义，
// 跨出去就违反支配关系。顺序不靠这条 SSA 边——`afir.sum` 对 destination 既读又写。
// CHECK: afir.store %[[DST]]
// 携带方案（初稿的备选 B）会在这里留下 `iter_args`，实施中已推翻，最终 IR 里一个都没有。
// CHECK-NOT: iter_args

// ALLOC-LABEL: func.func private @reduce_sum_common
// 两块 scratch 都是**普通**缓冲（`buffer_desc` 没有 `slot_count`/`queue_depth`），各自一个
// TBuf，尺寸各自来自一个 tiling 字段。合并成一块、或把尺寸配反，ReduceSum 的临时区就会和
// 段结果互相覆盖——UB 是紧排的，越界落在邻居身上，未必当场报错。
// ALLOC: %[[B0:[a-zA-Z0-9_]+]] = afir.tiling_param "b0_size"
// ALLOC: %[[B1:[a-zA-Z0-9_]+]] = afir.tiling_param "b1_size"
// 队列先于 TBuf 发 `InitBuffer`：TPipe 按调用序线性分配 UB，scatter 型 API 的共享 tmp
// 必须位于队列槽之上（`CanonicalResources.cpp` 的两趟发射）。TBuf 之间的相对序仍是
// `resources` 的键序。
// ALLOC: %[[QOUT:[a-zA-Z0-9_]+]] = ascendc.queue : <vecout, 2>
// ALLOC: %[[TSEG:[a-zA-Z0-9_]+]] = ascendc.tbuf : <veccalc>
// ALLOC: ascendc.pipe.init_buffer %{{[a-zA-Z0-9_]+}}, %[[TSEG]], %[[B0]]
// ALLOC: %[[TAPI:[a-zA-Z0-9_]+]] = ascendc.tbuf : <veccalc>
// ALLOC: ascendc.pipe.init_buffer %{{[a-zA-Z0-9_]+}}, %[[TAPI]], %[[B1]]
// `(8 * Ceiling((Rational(1 , 8) * z1t_size)))`：sympy 把字面分数折进 `Rational`，
// index 侧下降必须认这个形状，否则不收敛——而且是**不收敛**，不是报错：下降器先建了算子
// 再返回失败，贪婪重写驱动于是反复振荡。
// ALLOC-DAG: arith.ceildivsi
// ALLOC-DAG: arith.mulsi_extended
// 两块 scratch 的取用点都在函数作用域，两层循环都在它下面——`life_time_axis_id` 是 -1，
// 整个 kernel 一份。落进循环里不改结果（取的是同一块 TBuf），但会每轮重复取一次。
// `scratch[0]` 是高阶 API 的共享临时区（b1_size），`scratch[1]` 存段结果（b0_size），
// 顺序是调度器 `CalcReduceTmpSize` 的契约。
// ALLOC-DAG: %[[API:[a-zA-Z0-9_]+]] = ascendc.tbuf.get_tensor %[[TAPI]]
// ALLOC-DAG: %[[SEG:[a-zA-Z0-9_]+]] = ascendc.tbuf.get_tensor %[[TSEG]]
// z0Tb 循环。
// ALLOC: scf.for
// 累加器的队列槽位锚在 z1T 循环**之前**：sum 在循环内写它，store 在循环外读它。
// 锚点若取「第一个成员的 sourceOwner」会落到 store 上，分配就排在循环之后，use 早于 def。
// ALLOC: %[[SLOT:[a-zA-Z0-9_]+]] = ascendc.que_bind.alloc_tensor %[[QOUT]]
// ALLOC: %[[ACC:[a-zA-Z0-9_]+]] = ascendc.reinterpret_cast %[[SLOT]]
// z1T 循环。
// ALLOC: scf.for
// ALLOC: afir.load
// ALLOC: afir.sum %{{[a-zA-Z0-9_]+}} outs(%[[ACC]]) scratch(%[[API]], %[[SEG]])
// ALLOC: } {afir.axis_binding = #afir.axis_binding<axis = 4, role = loop>}
// ALLOC: afir.store %[[ACC]]
// ALLOC: ascendc.que_bind.free_tensor %[[QOUT]], %[[SLOT]]
// ALLOC-NOT: memref.alloc

// SELECT-LABEL: func.func private @reduce_sum_common
// 搬运选得出实现。这一行钉住两件事：`TransferPlan` 接受了两根向量化轴（二维 tile）、
// 它能在**第二层**循环里定位 tile。任何一条不成立，这里都会变成「找不到实现」。
//
// 实现 id 只有 `load.gm_ub` 一个：`DataCopyPadExtend` 从前是一档 target feature、选不中
// 时回落裸 `DataCopy`，现在一律发前者（与 legacy 的 `load_api_call.cpp` 一致），那一档
// 连同它的第二个 implementation 一并没有了。
// SELECT: afir.load
// SELECT-SAME: id = "load.gm_ub"
// 归约本身选到二维的 AR/RA 实现。发射出来的形状由
// `../conversion/afir-to-ascendc-reduce-sum.mlir` 钉住，这里只钉「选得出来」。
// SELECT: afir.sum
// SELECT-SAME: id = "reduce_sum.tensor_ar_ra"
