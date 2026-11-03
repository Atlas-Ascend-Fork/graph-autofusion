// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.lowered.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=fused_graph %t.lowered.mlir | FileCheck %s

// 符号形态的 `IndexExpr` 降到 AscendC。**与常量标量分得开的就是这一段**：
// `afir-scalar-binary-api.mlir` 那条的标量是一个 `arith.constant`，这条的标量是尺寸序言
// 里算出来的——`afir.tiling_param` 读一个运行期尺寸，`afir.symbolic_eval` 求值，再转成
// 元素类型。
//
// `symbolic_eval` 的表达式按 `__arg0` 引用操作数（自由符号在物化时被归一化过）。
// 到浮点要经 i64 中转，因为 `sitofp` 不收 `index`。**这两次转换漏掉或写反，IR 与生成的
// C++ 都还成立，只是算出另一个数**，所以下面逐个钉。
//
// 它**不占 kernel 形参**——这是 `IndexExpr` 与 `ScalarData` 的分界，签名那一行钉住它。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#output_desc = #afir.tensor_desc<axis = [0], role = <output>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>
#scalar_desc = #afir.tensor_desc<role = <input>>
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
#load_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>
#output_queue = #afir.buffer_desc<space = <ub>, buffer_id = 3, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.buffer_desc = #gm_buffer},
      %output: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#input_desc, #output_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 10,
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    // 尺寸序言：自由符号取成 `afir.tiling_param`，表达式求值，再转成元素类型。
    %size = afir.tiling_param "s0" : index
    %scaled = afir.symbolic_eval %size {expr = #afir.expr<"(2 * __arg0)">} : (index) -> index
    %wide = arith.index_cast %scaled : index to i64
    %scalar = arith.sitofp %wide : i64 to f16
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
      afir.operand_descs = [#compute_desc, #scalar_desc],
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

// CHECK: // -----KERNEL_BODY-----
// CHECK: __aicore__ void plan0_stage0_template0
// 序言里**读 TilingData 的字段**，不是把它折成常量。折成常量的话这一行就没有了。
// CHECK: {{v[0-9]+}} = {{v[0-9]+}}.s0;
// 两次转换，相邻两行：index→i64→half。少一次或写反都还编得过，只是算出另一个数。
// CHECK: int64_t {{v[0-9]+}} = static_cast<int64_t>(
// CHECK-NEXT: half {{v[0-9]+}} = static_cast<half>(
// CHECK: AscendC::Adds
// **签名里没有多出来的标量形参**：只有两个 GM 指针加 workspace 与 tiling。
// CHECK: extern "C"  __global__ __aicore__ void fused_graph(__gm__ half* {{v[0-9]+}}, __gm__ half* {{v[0-9]+}}, __gm__ uint8_t*
// CHECK: // -----AUTOFUSE_LAUNCH-----
// CHECK: extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, void* output0, void* workspace,
