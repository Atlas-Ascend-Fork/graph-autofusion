// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-cache-dependencies %s | FileCheck %s

// 带缓存调度的算子吃标量字面量时，字面量不带任何源轴：它的 `cache_refresh_axes` 只由其余
// 输入决定，全是循环不变量时为空数组，整个循环只算一次。有符号标量是 `arith.constant`，
// ui16 / ui32 标量是 bridge 建的 `afir.scalar`，两者必须一视同仁。只认前者的那版实现
// 会把后者当成未知输入，退化成每轮刷新（`array<i64: 3>`）。两个函数除常量写法外同形。

#gm = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"256">, life_time_axis_id = -1>
#buf = #afir.buffer_desc<space = <ub>, size_bytes_expr = <"256">, alignment_bytes = 32, life_time_axis_id = -1>
#x = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"1">, <"8">], strides = [<"8">, <"0">, <"1">], vectorized_axis = [2], vectorized_strides = [<"1">], role = <compute>>
#constant = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"1">, <"1">, <"8">], strides = [<"0">, <"0">, <"1">], vectorized_axis = [2], vectorized_strides = [<"1">], role = <compute>>
#sched = #afir.sched<axis = [3, 2], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>
#maps = [affine_map<(d0,d1,d2) -> (d0,d1,d2)>, affine_map<(d0,d1,d2) -> (d0,d1,d2)>]
#axes = #afir.asc_graph<tiling_key = 0, axes = [
    <id = 0, name = "r0", axis_type = Original, size = <"2">>,
    <id = 1, name = "r1", axis_type = Original, size = <"4">>,
    <id = 2, name = "at", axis_type = TileInner, size = <"8">, from = [4], split_pair_other_id = 5>,
    <id = 3, name = "r0r1", axis_type = Merged, size = <"8">, from = [0, 1]>,
    <id = 4, name = "a", axis_type = Original, size = <"8">>,
    <id = 5, name = "aT", axis_type = TileOuter, size = <"1">, from = [4], split_pair_other_id = 2>], type = Compute>
#scalar = #afir.tensor_desc<role = <input>>

module {
  func.func private @signed_scalar(%input: memref<2x4x8xf32> {afir.buffer_desc = #gm}, %tensor: tensor<2x4xf32> {afir.buffer_desc = #gm}) attributes {
    afir.function_role = #afir.function_role<kernel_body>,
    afir.arg_descs = [#x, #scalar],
    afir.asc_graph = #axes
  } {
    %reduce = afir.max %tensor {axis = 0 : i32} : tensor<2x4xf32> -> tensor<4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %tile = memref.alloc() {afir.buffer_desc = #buf} : memref<2x4x8xf32>
    %scaled = memref.alloc() {afir.buffer_desc = #buf} : memref<2x4x8xf32>
    %k = arith.constant 1.000000e+00 : f32
    %tile0 = afir.load %input outs(%tile) {afir.operand_descs = [#constant], afir.result_descs = [#constant], afir.destination_descs = [#constant], afir.sched = #sched, indexing_maps = #maps} : memref<2x4x8xf32>, memref<2x4x8xf32> -> memref<2x4x8xf32>
    %scaled0 = afir.add %tile0, %k outs(%scaled) {afir.operand_descs = [#constant, #scalar], afir.result_descs = [#constant], afir.destination_descs = [#constant], afir.sched = #sched} : (memref<2x4x8xf32>, f32), memref<2x4x8xf32> -> memref<2x4x8xf32>
    scf.for %r = %c0 to %c8 step %c1 {
      %x = afir.load %input outs(%tile) {afir.operand_descs = [#x], afir.result_descs = [#x], afir.destination_descs = [#x], afir.sched = #sched, indexing_maps = #maps} : memref<2x4x8xf32>, memref<2x4x8xf32> -> memref<2x4x8xf32>
    } {afir.axis_binding = #afir.axis_binding<axis = 3, role = loop>}
    return
  }

  func.func private @unsigned_scalar(%input: memref<2x4x8xui32> {afir.buffer_desc = #gm}, %tensor: tensor<2x4xf32> {afir.buffer_desc = #gm}) attributes {
    afir.function_role = #afir.function_role<kernel_body>,
    afir.arg_descs = [#x, #scalar],
    afir.asc_graph = #axes
  } {
    %reduce = afir.max %tensor {axis = 0 : i32} : tensor<2x4xf32> -> tensor<4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %tile = memref.alloc() {afir.buffer_desc = #buf} : memref<2x4x8xui32>
    %scaled = memref.alloc() {afir.buffer_desc = #buf} : memref<2x4x8xui32>
    %k = afir.scalar {value = 1 : ui32} : ui32
    %tile0 = afir.load %input outs(%tile) {afir.operand_descs = [#constant], afir.result_descs = [#constant], afir.destination_descs = [#constant], afir.sched = #sched, indexing_maps = #maps} : memref<2x4x8xui32>, memref<2x4x8xui32> -> memref<2x4x8xui32>
    %scaled0 = afir.add %tile0, %k outs(%scaled) {afir.operand_descs = [#constant, #scalar], afir.result_descs = [#constant], afir.destination_descs = [#constant], afir.sched = #sched} : (memref<2x4x8xui32>, ui32), memref<2x4x8xui32> -> memref<2x4x8xui32>
    scf.for %r = %c0 to %c8 step %c1 {
      %x = afir.load %input outs(%tile) {afir.operand_descs = [#x], afir.result_descs = [#x], afir.destination_descs = [#x], afir.sched = #sched, indexing_maps = #maps} : memref<2x4x8xui32>, memref<2x4x8xui32> -> memref<2x4x8xui32>
    } {afir.axis_binding = #afir.axis_binding<axis = 3, role = loop>}
    return
  }
}

// CHECK-LABEL: func.func private @signed_scalar
// CHECK: afir.max %{{.*}} {axis = 0 : i32}
// CHECK: afir.load {{.*}}afir.cache_refresh_axes = array<i64>,
// CHECK: afir.add {{.*}}afir.cache_refresh_axes = array<i64>,
// CHECK: scf.for
// CHECK: afir.load {{.*}}afir.cache_refresh_axes = array<i64: 0>

// CHECK-LABEL: func.func private @unsigned_scalar
// CHECK: afir.max %{{.*}} {axis = 0 : i32}
// CHECK: afir.load {{.*}}afir.cache_refresh_axes = array<i64>,
// CHECK: afir.add {{.*}}afir.cache_refresh_axes = array<i64>,
// CHECK: scf.for
// CHECK: afir.load {{.*}}afir.cache_refresh_axes = array<i64: 0>
