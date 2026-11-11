// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s

// 标量操作数过整条下降流水：循环生成、内存分配、队列绑定。
//
// `afir-scalar-binary-api.mlir` 那份 IR 停在 `convert` 那一层，碰不到这些 pass。而这
// 几层每一层都默认「op 的每个操作数都是一块缓冲」：
//
//   - `CanonicalBindingCollector::recordArray` 要求每个操作数在定义点有
//     `afir.buffer_desc`；
//   - `CanonicalFunctionMemoryAllocator::rewriteCanonicalOperation` 要把每个操作数从
//     memref 换成物化出来的 `!ascendc.local_tensor`。
//
// 标量两条都不满足，而且两条的报错都跟标量没关系（「value has no buffer_desc」/
// 「no materialized AscendC value」）。这份用例就是为了让它们报出来——加它之前这两条
// 路是坏的，`afir-scalar-binary-api.mlir` 全绿。
//
// `Scheduled` 那一支不在这里：这份 IR 的 add 走的仍是 identity plan，
// 元素个数取自 `local_tensor.get_size`。标量形态走到那一支的是 CPU E2E
// （`load_add_scalar_store` / `load_scalar_add_store`）——原先这里写「标量形态按设计没有
// E2E」，那是跟着 L-1「ATT tiling 拒收含 Scalar 节点的图」写的，而 L-1 是假的。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#output_desc = #afir.tensor_desc<axis = [0], role = <output>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
#load_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>
#output_queue = #afir.buffer_desc<space = <ub>, buffer_id = 3, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(%input: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.buffer_desc = #gm_buffer}, %output: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#input_desc, #output_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 10,
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %scalar = arith.constant 2.000000e+00 : f16
    %load_tile = memref.alloc() {afir.buffer_desc = #load_buffer} : memref<16xf16>
    %compute_tile = memref.alloc() {afir.buffer_desc = #output_queue} : memref<16xf16>
    %loaded = afir.load %input outs(%load_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#input_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed = afir.add %loaded, %scalar outs(%compute_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#compute_desc, #compute_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, f16), memref<16xf16> -> memref<16xf16>
    afir.store %computed outs(%output) {
      afir.destination_descs = [#output_desc],
      afir.operand_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// 常量物化在循环外——它在整条循环里是同一个值。
// CHECK: %[[SCALAR:.*]] = arith.constant 2.000000e+00 : f16
// CHECK: scf.for
// 标量原样进调用的第三个位置：没有被物化成一块 local tensor，也没有被 reinterpret。
// CHECK: ascendc.adds_l2 %{{[0-9a-z]+}}, %{{[0-9a-z]+}}, %[[SCALAR]], %{{[0-9a-z]+}}
// CHECK-NOT: afir.add
