// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate %s | FileCheck %s --check-prefix=LOOP

// **循环建在第一个 AFIR Op 的位置上，所以排在它后面的循环不变值必须被提上去。**
//
// bridge 按 AscGraph 的节点次序就地建 `Scalar` / `IndexExpr` 的 `arith.constant`。节点
// 排在第一个搬运之前时（只有一个 Load 的图通常如此）常量正好落在循环之前，看不出问题；
// 排在后面时——比如掩码由图里现算、标量节点夹在两个计算之间——常量就留在了 `scf.for`
// **之后**，而用它的 Op 已经被搬进循环体。MLIR 的 dominance 校验当场拒收：
// `operand #N does not dominate this use`。
//
// 这份用例把常量**故意写在第一个 `afir.load` 之后**，复现的就是那个位置。
// `hoistLoopInvariantSetup` 此前只认 `memref.alloc` 那一条链，把种子改成「操作数的定义方
// 不是调度出去的 Op」之后这一类一起覆盖。
//
// E2E 侧的两条对应用例是 `load_select_scalar_store` 与 `load_where_scalar_store`；
// 这里钉的是结构，不用编 kernel 就能变红。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#output_desc = #afir.tensor_desc<axis = [0], role = <output>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"32">, life_time_axis_id = -1>
#ub_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"32">, alignment_bytes = 32, life_time_axis_id = -1>
#ub_buffer_1 = #afir.buffer_desc<space = <ub>, buffer_id = 3, slot_count = 2, queue_depth = 2, size_bytes_expr = <"32">, alignment_bytes = 32, life_time_axis_id = -1>

module {
  func.func private @scalar_defined_after_first_load(
      %input: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                              afir.buffer_desc = #gm_buffer},
      %output: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>,
                               afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#input_desc, #output_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"16">>],
      type = Compute>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %load_tile = memref.alloc() {afir.buffer_desc = #ub_buffer} : memref<16xf16>
    %loaded = afir.load %input outs(%load_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#input_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    // **位置就是本用例的全部理由**：这一行在第一个 AFIR Op 之后。
    %scalar = arith.constant 2.000000e+00 : f16
    %compute_tile = memref.alloc() {afir.buffer_desc = #ub_buffer_1} : memref<16xf16>
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

// LOOP-LABEL: func.func private @scalar_defined_after_first_load
// 常量排在循环**之前**，而用它的 Op 在循环**之内**。两条 CHECK 的先后就是判据：
// 不提升的话常量会打印在 `scf.for` 之后，第二条无处可匹配。
// LOOP: %[[SCALAR:.*]] = arith.constant 2.000000e+00 : f16
// LOOP: scf.for
// LOOP: afir.add %{{.*}}, %[[SCALAR]]
