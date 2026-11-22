// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-loop-generate %s | FileCheck %s
// The queue copy reads the completed accumulator once, after all three tiles.
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#desc0 = #afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"1">, <"1">, <"1">, <"1">, <"1">],strides = [<"0">, <"0">, <"1">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"0">, <"0">],role = <compute>>
#desc1 = #afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"1">, <"1">, <"1">, <"3">, <"8">],strides = [<"0">, <"0">, <"24">, <"8">, <"1">],vectorized_axis = [3, 5],vectorized_strides = [<"0">, <"1">],role = <compute>>
#desc2 = #afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"1">, <"1">, <"1">, <"1">, <"1">],strides = [<"0">, <"0">, <"1">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"0">, <"0">],role = <output>>
func.func private @final_reduce_copy(%workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}, %tiling: !emitasc.py_struct<"TilingData", [i64], ["block_dim"]> {afir.kernel_arg = #afir.kernel_arg<role = tiling>}) attributes {
    afir.function_role = #afir.function_role<kernel_body>, afir.arg_descs = [], afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [<id = 6, name = "a6", axis_type = Original, size = <"1">, align = <"1">>, <id = 7, name = "a7", axis_type = Original, size = <"2">, align = <"1">>, <id = 4, name = "a4", axis_type = Original, size = <"3">, align = <"1">>, <id = 3, name = "a3", axis_type = Original, size = <"1">, align = <"1">>, <id = 5, name = "a5", axis_type = Original, size = <"8">, align = <"1">>], type = Compute>} {
    %src = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"128">, life_time_axis_id = -1>} : memref<1x1x1x3x8xf32, #afir.space<ub>>
    %acc = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"128">, life_time_axis_id = -1>} : memref<1x1x1x1x1xf32, #afir.space<ub>>
    %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"128">, life_time_axis_id = -1>} : memref<128xui8, #afir.space<ub>>
    %round = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3, size_bytes_expr = <"128">, life_time_axis_id = -1>} : memref<128xui8, #afir.space<ub>>
    %dst = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 4, size_bytes_expr = <"128">, life_time_axis_id = -1>} : memref<1x1x1x1x1xf32, #afir.space<ub>>
    %gm = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"128">, life_time_axis_id = -1>} : memref<1x1x1x1x1xf32, #afir.space<gm>>
    %min = afir.min %src outs(%acc) scratch(%tmp, %round) {afir.destination_descs = [#desc0], afir.operand_descs = [#desc1], afir.result_descs = [#desc0], afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>, afir.scratch_roles = ["api_scratch", "round_result"], afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">]} : memref<1x1x1x3x8xf32, #afir.space<ub>>, memref<1x1x1x1x1xf32, #afir.space<ub>>, scratch(memref<128xui8, #afir.space<ub>>, memref<128xui8, #afir.space<ub>>) -> memref<1x1x1x1x1xf32, #afir.space<ub>>
    %copy = afir.ub2ub %min outs(%dst) {afir.destination_descs = [#desc0], afir.operand_descs = [#desc0], afir.result_descs = [#desc0], afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">]} : memref<1x1x1x1x1xf32, #afir.space<ub>>, memref<1x1x1x1x1xf32, #afir.space<ub>> -> memref<1x1x1x1x1xf32, #afir.space<ub>>
    afir.store %copy outs(%gm) {afir.destination_descs = [#desc2], afir.operand_descs = [#desc0], afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 7>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">], indexing_maps = [#map1, #map1]} : memref<1x1x1x1x1xf32, #afir.space<ub>>, memref<1x1x1x1x1xf32, #afir.space<gm>>
    return
}
// CHECK: afir.min {{.*}} outs(%[[ACC:[a-zA-Z0-9_]+]])
// CHECK-NEXT: } {afir.axis_binding = #afir.axis_binding<axis = 4, role = loop>}
// CHECK-NEXT: %[[COPY:[a-zA-Z0-9_]+]] = afir.ub2ub %[[ACC]]
// CHECK-NEXT: afir.store %[[COPY]]
