// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-cache-dependencies %s | FileCheck %s --check-prefix=DEPS
// RUN: af-opt --afir-cache-dependencies --afir-exec-condition %s | FileCheck %s --check-prefix=GUARD
// The merged R loop has two source coordinates: r/4 and r%4. The first
// Load refreshes every four iterations, the second every iteration. A value
// depending on both must refresh when either changes; a constant tile only
// refreshes at loop entry. The kept vector axis is unrelated to both R axes.
#gm = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"256">, life_time_axis_id = -1>
#buf = #afir.buffer_desc<space = <ub>, size_bytes_expr = <"256">, alignment_bytes = 32, life_time_axis_id = -1>
#x = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"1">, <"8">], strides = [<"8">, <"0">, <"1">], vectorized_axis = [2], vectorized_strides = [<"1">], role = <compute>>
#b = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"1">, <"4">, <"1">], strides = [<"0">, <"1">, <"0">], vectorized_axis = [2], vectorized_strides = [<"0">], role = <compute>>
#constant = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"1">, <"1">, <"8">], strides = [<"0">, <"0">, <"1">], vectorized_axis = [2], vectorized_strides = [<"1">], role = <compute>>
#full = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"4">, <"8">], strides = [<"32">, <"8">, <"1">], vectorized_axis = [2], vectorized_strides = [<"1">], role = <compute>>
#sched = #afir.sched<axis = [3, 2], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>
module {
func.func @reduce_cache(%input: memref<2x4x8xf32> {afir.buffer_desc = #gm}, %tensor: tensor<2x4xf32> {afir.buffer_desc = #gm}) attributes {
  afir.arg_descs = [#x, #afir.tensor_desc<role = <input>>],
  afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
    <id = 0, name = "r0", axis_type = Original, size = <"2">>,
    <id = 1, name = "r1", axis_type = Original, size = <"4">>,
    <id = 2, name = "at", axis_type = TileInner, size = <"8">, from = [4], split_pair_other_id = 5>,
    <id = 3, name = "r0r1", axis_type = Merged, size = <"8">, from = [0, 1]>,
    <id = 4, name = "a", axis_type = Original, size = <"8">>,
    <id = 5, name = "aT", axis_type = TileOuter, size = <"1">, from = [4], split_pair_other_id = 2>], type = Compute>
} {
  %reduce = afir.max %tensor {axis = 0 : i32} : tensor<2x4xf32> -> tensor<4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %dst0 = memref.alloc() {afir.buffer_desc = #buf} : memref<2x4x8xf32>
  %dst1 = memref.alloc() {afir.buffer_desc = #buf} : memref<2x4x8xf32>
  %dst2 = memref.alloc() {afir.buffer_desc = #buf} : memref<2x4x8xf32>
  %dst3 = memref.alloc() {afir.buffer_desc = #buf} : memref<2x4x8xf32>
  %dst4 = memref.alloc() {afir.buffer_desc = #buf} : memref<2x4x8xf32>
  scf.for %r = %c0 to %c8 step %c1 {
    %outer = arith.divui %r, %c4 {afir.axis_binding = #afir.axis_binding<axis = 0, role = axis_index>} : index
    %inner = arith.remui %r, %c4 {afir.axis_binding = #afir.axis_binding<axis = 1, role = axis_index>} : index
    %x = afir.load %input outs(%dst0) {afir.operand_descs = [#x], afir.result_descs = [#x], afir.destination_descs = [#x], afir.sched = #sched, indexing_maps = [affine_map<(d0,d1,d2) -> (d0,d1,d2)>, affine_map<(d0,d1,d2) -> (d0,d1,d2)>]} : memref<2x4x8xf32>, memref<2x4x8xf32> -> memref<2x4x8xf32>
    %b = afir.load %input outs(%dst1) {afir.operand_descs = [#b], afir.result_descs = [#b], afir.destination_descs = [#b], afir.sched = #sched, indexing_maps = [affine_map<(d0,d1,d2) -> (d0,d1,d2)>, affine_map<(d0,d1,d2) -> (d0,d1,d2)>]} : memref<2x4x8xf32>, memref<2x4x8xf32> -> memref<2x4x8xf32>
    %c = afir.load %input outs(%dst2) {afir.operand_descs = [#constant], afir.result_descs = [#constant], afir.destination_descs = [#constant], afir.sched = #sched, indexing_maps = [affine_map<(d0,d1,d2) -> (d0,d1,d2)>, affine_map<(d0,d1,d2) -> (d0,d1,d2)>]} : memref<2x4x8xf32>, memref<2x4x8xf32> -> memref<2x4x8xf32>
    %abs = afir.abs %x outs(%dst3) {afir.operand_descs = [#x], afir.result_descs = [#x], afir.destination_descs = [#x], afir.sched = #sched} : memref<2x4x8xf32>, memref<2x4x8xf32> -> memref<2x4x8xf32>
    %add = afir.add %abs, %b outs(%dst4) {afir.operand_descs = [#x, #b], afir.result_descs = [#full], afir.destination_descs = [#full], afir.sched = #sched} : (memref<2x4x8xf32>, memref<2x4x8xf32>), memref<2x4x8xf32> -> memref<2x4x8xf32>
  } {afir.axis_binding = #afir.axis_binding<axis = 3, role = loop>}
  return
}
}
// DEPS: afir.load {{.*}}afir.cache_refresh_axes = array<i64: 0>
// DEPS: afir.load {{.*}}afir.cache_refresh_axes = array<i64: 1>
// DEPS: afir.load {{.*}}afir.cache_refresh_axes = array<i64>
// DEPS: afir.abs {{.*}}afir.cache_refresh_axes = array<i64: 0>
// DEPS: afir.add {{.*}}afir.cache_refresh_axes = array<i64: 0, 1>
// GUARD: scf.for
// GUARD: %[[OUTER:.*]] = arith.divui
// GUARD: %[[INNER:.*]] = arith.remui
// GUARD: %[[PREV_OUTER:.*]] = arith.divui
// GUARD: arith.cmpi ne, %[[OUTER]], %[[PREV_OUTER]]
// GUARD: scf.if
// GUARD: afir.load
// GUARD: %[[PREV_INNER:.*]] = arith.remui
// GUARD: arith.cmpi ne, %[[INNER]], %[[PREV_INNER]]
// GUARD: scf.if
// GUARD: afir.load
// GUARD: %[[FIRST:.*]] = arith.cmpi eq
// GUARD-NOT: arith.cmpi ne
// GUARD: scf.if %[[FIRST]]
// GUARD: afir.load
// GUARD: scf.if
// GUARD: afir.abs
// GUARD: arith.cmpi ne, %[[OUTER]]
// GUARD: arith.cmpi ne, %[[INNER]]
// GUARD: scf.if
// GUARD: afir.add
