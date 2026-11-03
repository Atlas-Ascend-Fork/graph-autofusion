// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt %s | FileCheck %s

#desc_1d = #afir.tensor_desc<axis = [0], role = #afir.role<compute>>
#desc_2d = #afir.tensor_desc<axis = [0, 1], role = #afir.role<compute>>
#buffer = #afir.buffer_desc<space = #afir.space<ub>,>

func.func @logical_load_store(%input: tensor<4xf16>) -> tensor<4xf16> {
  %loaded = afir.load %input : tensor<4xf16> -> tensor<4xf16>
  %stored = afir.store %loaded : tensor<4xf16> -> tensor<4xf16>
  return %stored : tensor<4xf16>
}

// CHECK-LABEL: func.func @logical_load_store
// CHECK: afir.load %arg0 : tensor<4xf16> -> tensor<4xf16>
// CHECK: afir.store %{{.*}} : tensor<4xf16> -> tensor<4xf16>

func.func @buffer_load_store(%input: memref<4xf16> {afir.buffer_desc = #buffer}, %output: memref<4xf16> {afir.buffer_desc = #buffer}) attributes {
  afir.arg_descs = [#desc_1d, #desc_1d]
} {
  %loaded = afir.load %input outs(%output) {
    afir.operand_descs = [#desc_1d],
    afir.result_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  afir.store %loaded outs(%output) {
    afir.operand_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16>
  return
}

// CHECK-LABEL: func.func @buffer_load_store
// CHECK: afir.load %{{.*}} outs(%{{.*}})
// CHECK-SAME: indexing_maps = [#{{.*}}, #{{.*}}]
// CHECK-SAME: : memref<4xf16>, memref<4xf16> -> memref<4xf16>
// CHECK: afir.store %{{.*}} outs(%{{.*}})
// CHECK-SAME: indexing_maps = [#{{.*}}, #{{.*}}]
// CHECK-SAME: : memref<4xf16>, memref<4xf16>

func.func @buffer_add(%lhs: memref<4xf16> {afir.buffer_desc = #buffer}, %rhs: memref<4xf16> {afir.buffer_desc = #buffer}, %output: memref<4xf16> {afir.buffer_desc = #buffer}) attributes {
  afir.arg_descs = [#desc_1d, #desc_1d, #desc_1d]
} {
  %sum = afir.add %lhs, %rhs outs(%output) {
    afir.operand_descs = [#desc_1d, #desc_1d],
    afir.result_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d]
  } : (memref<4xf16>, memref<4xf16>), memref<4xf16> -> memref<4xf16>
  return
}

// CHECK-LABEL: func.func @buffer_add
// CHECK: afir.add
// CHECK-SAME: : (memref<4xf16>, memref<4xf16>), memref<4xf16> -> memref<4xf16>

func.func @buffer_add_broadcast(%lhs: memref<1x4xf16> {afir.buffer_desc = #buffer}, %rhs: memref<2x4xf16> {afir.buffer_desc = #buffer}, %output: memref<2x4xf16> {afir.buffer_desc = #buffer}) attributes {
  afir.arg_descs = [#desc_2d, #desc_2d, #desc_2d]
} {
  %sum = afir.add %lhs, %rhs outs(%output) {
    afir.operand_descs = [#desc_2d, #desc_2d],
    afir.result_descs = [#desc_2d],
    afir.destination_descs = [#desc_2d]
  } :
    (memref<1x4xf16>, memref<2x4xf16>), memref<2x4xf16> -> memref<2x4xf16>
  return
}

// CHECK-LABEL: func.func @buffer_add_broadcast
// CHECK: afir.add
// CHECK-SAME: : (memref<1x4xf16>, memref<2x4xf16>), memref<2x4xf16> -> memref<2x4xf16>

func.func @buffer_canonical_element_types(%input: memref<4xi8> {afir.buffer_desc = #buffer}, %destination: memref<4xi8> {afir.buffer_desc = #buffer}, %lhs: memref<4xi16> {afir.buffer_desc = #buffer}, %rhs: memref<4xi16> {afir.buffer_desc = #buffer}, %output: memref<4xi16> {afir.buffer_desc = #buffer}) attributes {
  afir.arg_descs = [#desc_1d, #desc_1d, #desc_1d, #desc_1d, #desc_1d]
} {
  %loaded = afir.load %input outs(%destination) {
    afir.operand_descs = [#desc_1d],
    afir.result_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xi8>, memref<4xi8> -> memref<4xi8>
  %sum = afir.add %lhs, %rhs outs(%output) {
    afir.operand_descs = [#desc_1d, #desc_1d],
    afir.result_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d]
  } :
    (memref<4xi16>, memref<4xi16>), memref<4xi16> -> memref<4xi16>
  afir.store %loaded outs(%destination) {
    afir.operand_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xi8>, memref<4xi8>
  return
}

// CHECK-LABEL: func.func @buffer_canonical_element_types
// CHECK: memref<4xi8>
// CHECK: memref<4xi16>

func.func @rank_changing_transfer(%input: memref<2x3xf16> {afir.buffer_desc = #buffer}, %tile: memref<6xf16> {afir.buffer_desc = #buffer}) attributes {
  afir.arg_descs = [#desc_2d, #desc_1d]
} {
  %loaded = afir.load %input outs(%tile) {
    afir.operand_descs = [#desc_2d],
    afir.result_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d],
    indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0 * 3 + d1)>]
  } : memref<2x3xf16>, memref<6xf16> -> memref<6xf16>
  afir.store %loaded outs(%input) {
    afir.operand_descs = [#desc_1d],
    afir.destination_descs = [#desc_2d],
    indexing_maps = [affine_map<(d0, d1)[s0] -> (d0 * 3 + d1)>,
                     affine_map<(d0, d1)[s0] -> (d0, d1)>]
  } : memref<6xf16>, memref<2x3xf16>
  return
}

// CHECK-LABEL: func.func @rank_changing_transfer
// CHECK: indexing_maps = [#{{.*}}, #{{.*}}]
// CHECK: afir.store

func.func @load_add_store_pipeline(%lhs_gm: memref<4xf16> {afir.buffer_desc = #buffer}, %rhs_gm: memref<4xf16> {afir.buffer_desc = #buffer}, %out_gm: memref<4xf16> {afir.buffer_desc = #buffer}, %lhs_tile: memref<4xf16> {afir.buffer_desc = #buffer}, %rhs_tile: memref<4xf16> {afir.buffer_desc = #buffer}, %out_tile: memref<4xf16> {afir.buffer_desc = #buffer}) attributes {
  afir.arg_descs = [#desc_1d, #desc_1d, #desc_1d, #desc_1d, #desc_1d, #desc_1d]
} {
  %lhs = afir.load %lhs_gm outs(%lhs_tile) {
    afir.operand_descs = [#desc_1d],
    afir.result_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  %rhs = afir.load %rhs_gm outs(%rhs_tile) {
    afir.operand_descs = [#desc_1d],
    afir.result_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  %sum = afir.add %lhs, %rhs outs(%out_tile) {
    afir.operand_descs = [#desc_1d, #desc_1d],
    afir.result_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d]
  } :
    (memref<4xf16>, memref<4xf16>), memref<4xf16> -> memref<4xf16>
  afir.store %sum outs(%out_gm) {
    afir.operand_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16>
  return
}

// CHECK-LABEL: func.func @load_add_store_pipeline
// CHECK: afir.load
// CHECK: afir.load
// CHECK: afir.add
// CHECK: afir.store

// 被调度过的 GM -> UB 搬运，**两侧形状都静态且元素数不等**，仍然合法。
//
// GM 侧是整块张量的视图，kernel 实参上连形状都不发（`memref<f32>` 是个裸指针）；UB 侧是一块
// tile。两者元素数本来就不等，对得上的是 descriptor，不是 memref 的元素数。这一档过去被
// 「元素数相等」那条规则误伤过：豁免只写在「有动态维」那一支里，于是归约轴不切分的候选
// （整条 R 一次进 UB，UB 缓冲因此是静态的）在建模块时就被验证器拒掉，报成
// `transfer source and destination must have the same static element count`。
#gm_scalar = #afir.buffer_desc<space = #afir.space<gm>, size_bytes_expr = #afir.expr<"4">, life_time_axis_id = -1>
#ub_tile = #afir.buffer_desc<space = #afir.space<ub>, buffer_id = 0, life_time_axis_id = -1>
#gm_view = #afir.tensor_desc<role = #afir.role<input>>
#ub_view = #afir.tensor_desc<axis = [4, 5, 3, 1],
                             repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1024">],
                             strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"1024">, #afir.expr<"1">],
                             vectorized_axis = [3, 1],
                             vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">],
                             role = #afir.role<compute>>

func.func private @scheduled_whole_reduce_axis_load(
    %gm: memref<f32, #afir.space<gm>> {afir.buffer_desc = #gm_scalar})
    attributes {afir.arg_descs = [#gm_view], afir.function_role = #afir.function_role<kernel_body>} {
  %tile = memref.alloc() {afir.buffer_desc = #ub_tile} : memref<1x1x1x1024xf32, #afir.space<ub>>
  %loaded = afir.load %gm outs(%tile) {
    afir.operand_descs = [#gm_view],
    afir.destination_descs = [#ub_view],
    afir.result_descs = [#ub_view],
    afir.sched = #afir.sched<axis = [4, 5, 3, 1], loop_axis = 5>,
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
  } : memref<f32, #afir.space<gm>>, memref<1x1x1x1024xf32, #afir.space<ub>>
    -> memref<1x1x1x1024xf32, #afir.space<ub>>
  return
}

// CHECK-LABEL: func.func private @scheduled_whole_reduce_axis_load
// CHECK: afir.load
