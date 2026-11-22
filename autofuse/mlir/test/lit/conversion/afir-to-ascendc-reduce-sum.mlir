// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.sum` 的下降。输入与 `../demo/afir_loop_generate_reduce_accumulation.mlir` 是同一份
// 真实调度（S0 通用模板，`f32[16, 32]` 归约末维），这里把流水线跑到底。
//
// 展开的形状与 legacy 逐调用对齐（`codegen/api_call/reduce/reduce_api_call.cpp:86` 起的
// multi-reduce 分支）：每个 z1T 轮次一次 `ReduceInit`、一次 `ReduceSum`、两个 `PipeBarrier`，
// 再按「是不是第一轮」拷贝或累加。

// 两个目标各生成一遍（950 无 E2E，只能比对文本）：reduce 的下降没有任何目标相关分支，
// 两份输出**逐字节相同**，所以共用同一组 CHECK。这是
// `2026-09-20-two-dim-transfer-addressing-design.md` 8.5 六组合验收项的 lit 一侧。
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6)>
module @reduce_sum_lowering attributes {afir.cube_type = -1 : i64, afir.enable_group_parallel = false, afir.is_reduce_mem_reuse = false, afir.module_kind = "plan", afir.workspace_tensor_ids = array<i64: 2>, afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>, afplan.stages = #afplan.stages<template_counts = [1]>, ascendc.target_aiv_num = 48 : i64, ascendc.target_ub_size = 196608 : i64} {
  func.func private @reduce_sum_lowering_body(%arg0: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}, %arg1: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>, #afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"16">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>, <id = 2, name = "z0T", axis_type = TileOuter, size = <"Ceiling((16 / (z0t_size)))">, align = <"1">, from = [0], split_pair_other_id = 3>, <id = 3, name = "z0t", axis_type = TileInner, size = <"z0t_size">, align = <"1">, from = [0], split_pair_other_id = 2>, <id = 4, name = "z1T", axis_type = TileOuter, size = <"Ceiling((32 / (z1t_size)))">, align = <"1">, from = [1], split_pair_other_id = 5>, <id = 5, name = "z1t", axis_type = TileInner, size = <"z1t_size">, align = <"1">, from = [1], split_pair_other_id = 4>, <id = 6, name = "z0TB", axis_type = BlockOuter, size = <"Ceiling((Ceiling((16 / (z0t_size))) / (z0Tb_size)))">, align = <"1">, from = [2], split_pair_other_id = 7>, <id = 7, name = "z0Tb", axis_type = BlockInner, size = <"z0Tb_size">, align = <"1">, from = [2], split_pair_other_id = 6>], type = Compute, size_var = [<"z0t_size">, <"z1t_size">, <"z0Tb_size">]>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_sum_store_16x32_0_B0Y0R1_S0G0C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>], afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
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

// CHECK-LABEL: func.func private @reduce_sum_lowering_body
// 两块 scratch 各自一个 TBuf，尺寸各自来自一个 tiling 字段。角色由 `ReducePlan` 的 scratch
// 顺序定：`scratch[0]`（b1_size）是高阶 API 的共享临时区，`scratch[1]`（b0_size）存段结果。
// 配反了 ReduceSum 的临时区就会踩掉段结果，而 UB 紧排，越界落在邻居身上未必当场报错。
// CHECK: %[[B0M:[a-zA-Z0-9_]+]] = emitasc.member %{{[a-zA-Z0-9_]+}} "b0_size"
// CHECK: %[[B0:[a-zA-Z0-9_]+]] = arith.index_cast %[[B0M]]
// CHECK: %[[B1M:[a-zA-Z0-9_]+]] = emitasc.member %{{[a-zA-Z0-9_]+}} "b1_size"
// CHECK: %[[B1:[a-zA-Z0-9_]+]] = arith.index_cast %[[B1M]]
// CHECK: %[[TSEG:[a-zA-Z0-9_]+]] = ascendc.tbuf : <veccalc>
// CHECK: ascendc.pipe.init_buffer %{{[a-zA-Z0-9_]+}}, %[[TSEG]], %[[B0]]
// CHECK: %[[TAPI:[a-zA-Z0-9_]+]] = ascendc.tbuf : <veccalc>
// CHECK: ascendc.pipe.init_buffer %{{[a-zA-Z0-9_]+}}, %[[TAPI]], %[[B1]]
// 取用点在函数作用域，两层循环都在它下面。
// CHECK-DAG: %[[API:[a-zA-Z0-9_]+]] = ascendc.tbuf.get_tensor %[[TAPI]]
// CHECK-DAG: %[[SEGBUF:[a-zA-Z0-9_]+]] = ascendc.tbuf.get_tensor %[[TSEG]]

// 累加器的队列槽位在被归约的 tile 循环之外分配一次。
// CHECK: %[[QOUT_SLOT:.*]] = ascendc.que_bind.alloc_tensor %{{.*}} : !ascendc.queue<vecout, 2>
// CHECK: %[[DST:.*]] = ascendc.reinterpret_cast %[[QOUT_SLOT]]

// z1T 循环：每轮搬一个 tile 进来，归约，再并进累加器。
// CHECK: scf.for %[[IV:[a-zA-Z0-9_]+]] =
// CHECK: ascendc.data_copy_pad_extend
// CHECK: ascendc.pipe_barrier pipe_mte2

// 对齐补出来的位置先填成 0，否则它们会被一起加进和里。`reduceType = 2` 是
// `reduce_init.h` 的 `kReduceOpSum`。填的是**输入**那块 tile，不是归约的落点。
//
// 第 4 个参数（`dim_r_current`）是段长对齐之后的值，第 5 个（`inner_r`）是原始段长：
// helper 用前者作整块填充的起点（必须 32B 对齐），用后者算每行最后一个 32B 块里的掩码。
// 传同一个值进去会让整块填充从一个不对齐的偏移开始，`Duplicate` 的地址检查当场 abort。
// 对齐两次：`%[[R]]` 是 tile 大小对齐后的 R（循环无关，也是 UB 里的行距），
// `%[[RCUR]]` 是本轮实际段长对齐后的值。
//
// 对齐的乘法走 `mulsi_extended` + `cf.assert`，不是裸 `muli`：这些量都来自 tiling、运行期
// 才知道值，裸乘溢出会静默回绕成一个小的正数，被当成长度用下去是算错值而不是报错。
// CHECK: %[[RDIV:[a-zA-Z0-9_]+]] = arith.ceildivsi %{{[a-zA-Z0-9_]+}}, %[[C8:[a-zA-Z0-9_]+]]
// CHECK: %[[R:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = arith.mulsi_extended %[[RDIV]], %[[C8]]
// CHECK: cf.assert %{{[a-zA-Z0-9_]+}}, "reduce aligned extent must not overflow"
// CHECK: %[[RCURDIV:[a-zA-Z0-9_]+]] = arith.ceildivsi %[[RACT:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}}
// CHECK: %[[RCUR:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = arith.mulsi_extended %[[RCURDIV]], %{{[a-zA-Z0-9_]+}}
// CHECK: cf.assert %{{[a-zA-Z0-9_]+}}, "reduce aligned extent must not overflow"
// CHECK: ascendc.local_tensor.subindex %[[SRC:[a-zA-Z0-9_]+]]
// CHECK: ascendc.duplicate_l1
// CHECK: ascendc.pipe_barrier pipe_v

// 段结果写进自己的 scratch（`%[[SEG]]`），不是直接写累加器——写进去就没法区分首轮了。
// layout = 0 是 AR：被归约的是末维。
// CHECK: %[[SEG:[a-zA-Z0-9_]+]] = ascendc.reinterpret_cast %[[SEGBUF]]
// CHECK: ascendc.reduce_sum_2d_l2 %[[SEG]], %[[SRC]], %[[API]], %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}
// CHECK-SAME: layout = 0
// `srcInnerPad` 恒为 true：CANN 对它是无条件断言，传 false 直接 abort。
// CHECK-SAME: srcInnerPad = true
// CHECK: ascendc.pipe_barrier pipe_v

// 首轮拷贝、其余轮次相加，与 legacy 的 `if (z1T == 0)` 是同一条判据。
// CHECK: %[[FIRST:.*]] = arith.cmpi eq, %[[IV]], %{{.*}}
// CHECK: scf.if %[[FIRST]] {
// CHECK:   ascendc.data_copy_l2 %[[DST]], %[[SEG]]
// CHECK: } else {
// CHECK:   ascendc.add_l2 %[[DST]], %[[DST]], %[[SEG]]
// CHECK: }

// 归约完成后才出队，Store 在循环之外读它。
// CHECK: } {afir.axis_binding = #afir.axis_binding<axis = 4, role = loop>}
// CHECK: ascendc.que_bind.enque_tensor
// CHECK: ascendc.data_copy_pad_extend
// CHECK-NOT: afir.sum
