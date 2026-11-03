// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --verify-diagnostics --split-input-file %s

func.func @missing_indexing_maps(%input: memref<4xf16>, %output: memref<4xf16>) {
  // expected-error @+1 {{InvalidIndexingMaps: buffer form requires an ArrayAttr of AffineMapAttr}}
  %loaded = afir.load %input outs(%output) : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

func.func @one_indexing_map(%input: memref<4xf16>, %output: memref<4xf16>) {
  // expected-error @+1 {{InvalidIndexingMaps: transfer requires exactly two indexing maps}}
  %loaded = afir.load %input outs(%output) {
    indexing_maps = [affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

func.func @three_indexing_maps(%input: memref<4xf16>, %output: memref<4xf16>) {
  // expected-error @+1 {{InvalidIndexingMaps: transfer requires exactly two indexing maps}}
  %loaded = afir.load %input outs(%output) {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

func.func @different_iteration_ranks(%input: memref<4xf16>, %output: memref<4xf16>) {
  // expected-error @+1 {{InvalidIndexingMaps: source and destination maps must share the same iteration rank}}
  %loaded = afir.load %input outs(%output) {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0, d1) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

func.func @load_source_map_rank_mismatch(%input: memref<2x3xf16>, %output: memref<6xf16>) {
  // expected-error @+1 {{InvalidIndexingMaps: source map result rank must match source view rank}}
  %loaded = afir.load %input outs(%output) {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0 * 3 + d1)>]
  } : memref<2x3xf16>, memref<6xf16> -> memref<6xf16>
  return
}

// -----

func.func @store_destination_map_rank_mismatch(%input: memref<6xf16>, %output: memref<2x3xf16>) {
  // expected-error @+1 {{InvalidIndexingMaps: destination map result rank must match destination view rank}}
  afir.store %input outs(%output) {
    indexing_maps = [affine_map<(d0, d1) -> (d0 * 3 + d1)>, affine_map<(d0, d1) -> (d0)>]
  } : memref<6xf16>, memref<2x3xf16>
  return
}

// -----

func.func @load_element_type_mismatch(%input: memref<4xi16>, %output: memref<4xf16>) {
  // expected-error @+1 {{transfer source and destination must have the same element type}}
  %loaded = afir.load %input outs(%output) {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xi16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

func.func @store_static_element_count_mismatch(%input: memref<4xf16>, %output: memref<8xf16>) {
  // expected-error @+1 {{transfer source and destination must have the same static element count}}
  afir.store %input outs(%output) {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<8xf16>
  return
}

// -----

func.func @different_mapping_symbol_counts(%input: memref<4xf16>, %output: memref<4xf16>) {
  // expected-error @+1 {{InvalidIndexingMaps: source and destination maps must share the same symbol count}}
  %loaded = afir.load %input outs(%output) {
    indexing_maps = [affine_map<(d0)[s0] -> (d0 + s0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

func.func @missing_canonical_dynamic_descriptors(%input: memref<?x3xf16>, %output: memref<?xf16>) {
  // expected-error @+1 {{InvalidTransferShape: missing canonical descriptors for dynamic transfer}}
  %loaded = afir.load %input outs(%output) {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0 * 3 + d1)>]
  } : memref<?x3xf16>, memref<?xf16> -> memref<?xf16>
  return
}

// -----

func.func @non_affine_map_attribute(%input: memref<4xf16>, %output: memref<4xf16>) {
  // expected-error @+1 {{attribute 'indexing_maps' failed to satisfy constraint: AffineMap array attribute}}
  %loaded = afir.load %input outs(%output) {
    indexing_maps = [1 : i32, 1 : i32]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

func.func @store_result_in_buffer_form(%input: memref<4xf16>, %output: memref<4xf16>) {
  // expected-error @+1 {{buffer form has an invalid result count}}
  %stored = afir.store %input outs(%output) {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

func.func @load_result_destination_mismatch(%input: memref<4xf16>, %output: memref<8xf16>) {
  // expected-error @+1 {{buffer result and destination must have the same type}}
  %loaded = afir.load %input outs(%output) {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<8xf16> -> memref<4xf16>
  return
}

// -----

func.func @add_result_destination_mismatch(%lhs: memref<4xf16>, %rhs: memref<4xf16>, %output: memref<8xf16>) {
  // expected-error @+1 {{buffer result and destination must have the same type}}
  %sum = afir.add %lhs, %rhs outs(%output) : (memref<4xf16>, memref<4xf16>), memref<8xf16> -> memref<4xf16>
  return
}

// -----

func.func @add_element_type_mismatch(
    %lhs: memref<4xi16>, %rhs: memref<4xf16>, %output: memref<4xf16>) {
  // 输入之间的 dtype 由 ODS trait 管，比 Op 的 verifier 更早；destination 与结果的
  // dtype 由上面那条「buffer result and destination must have the same type」管。
  // expected-error @+1 {{all of {lhs, rhs, result} have same element type}}
  %sum = afir.add %lhs, %rhs outs(%output) :
    (memref<4xi16>, memref<4xf16>), memref<4xf16> -> memref<4xf16>
  return
}

// -----

func.func @add_incompatible_input_shapes(
    %lhs: memref<2xf16>, %rhs: memref<3xf16>, %output: memref<3xf16>) {
  // expected-error @+1 {{operands could not be broadcast to a compatible shape}}
  %sum = afir.add %lhs, %rhs outs(%output) :
    (memref<2xf16>, memref<3xf16>), memref<3xf16> -> memref<3xf16>
  return
}

// -----

func.func @add_destination_shape_mismatch(
    %lhs: memref<1x4xf16>, %rhs: memref<2x4xf16>, %output: memref<4x4xf16>) {
  // expected-error @+1 {{result shape must match the inferred broadcast shape}}
  %sum = afir.add %lhs, %rhs outs(%output) :
    (memref<1x4xf16>, memref<2x4xf16>), memref<4x4xf16> -> memref<4x4xf16>
  return
}

// -----

func.func @load_unsupported_buffer_element_type(%input: memref<4xf64>, %output: memref<4xf64>) {
  // expected-error @+1 {{must be Tensor or MemRef of}}
  %loaded = afir.load %input outs(%output) {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf64>, memref<4xf64> -> memref<4xf64>
  return
}

// -----

func.func @store_unsupported_buffer_element_type(%input: memref<4xf64>, %output: memref<4xf64>) {
  // expected-error @+1 {{must be Tensor or MemRef of}}
  afir.store %input outs(%output) {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf64>, memref<4xf64>
  return
}

// -----

func.func @add_unsupported_buffer_element_type(
    %lhs: memref<4xf64>, %rhs: memref<4xf64>, %output: memref<4xf64>) {
  // expected-error @+1 {{must be Tensor or MemRef of}}
  %sum = afir.add %lhs, %rhs outs(%output) :
    (memref<4xf64>, memref<4xf64>), memref<4xf64> -> memref<4xf64>
  return
}

// -----

func.func @missing_canonical_buffer_descriptors(%input: memref<4xf16>, %output: memref<4xf16>) {
  // expected-error @+1 {{InvalidDescriptor: buffer form requires 'afir.operand_descs'}}
  %loaded = afir.load %input outs(%output) {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

// 规划结果只在值的定义点存一份，使用点不再带数组（设计文档 5.1 步 8）：定义点没有
// 就是没有，报在使用点上。
func.func @operand_without_definition_buffer_descriptor(%input: memref<4xf16>, %output: memref<4xf16>) {
  // expected-error @+1 {{InvalidDescriptor: 'afir.operand_descs' element 0 has no 'afir.buffer_desc' at its definition}}
  %loaded = afir.load %input outs(%output) {
    afir.operand_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<input>>],
    afir.result_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<compute>>],
    afir.destination_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<compute>>],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

func.func @missing_result_tensor_descriptor(%input: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<ub>, life_time_axis_id = -1>}, %output: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<ub>, life_time_axis_id = -1>}) {
  // expected-error @+1 {{InvalidDescriptor: buffer form requires 'afir.result_descs'}}
  %loaded = afir.load %input outs(%output) {
    afir.operand_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<input>>],
    afir.destination_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<compute>>],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

func.func @destination_without_definition_buffer_descriptor(
    %input: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<ub>, life_time_axis_id = -1>},
    %output: memref<4xf16>) {
  // expected-error @+1 {{InvalidDescriptor: 'afir.destination_descs' element 0 has no 'afir.buffer_desc' at its definition}}
  afir.store %input outs(%output) {
    afir.operand_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<compute>>],
    afir.destination_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<output>>],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16>
  return
}

// -----

func.func @descriptor_axis_rank_mismatch(%input: memref<4xf16>, %output: memref<4xf16>) {
  // expected-error @+1 {{InvalidDescriptor: 'afir.operand_descs' element 0 axis count 2 must match shaped rank 1}}
  %loaded = afir.load %input outs(%output) {
    afir.operand_descs = [#afir.tensor_desc<axis = [0, 1], role = #afir.role<input>>],
    afir.result_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<compute>>],
    afir.destination_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<compute>>],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}

// -----

// expected-error @+1 {{InvalidDescriptor: buffer form requires 'afir.arg_descs'}}
func.func @missing_argument_descriptors(%input: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<ub>, life_time_axis_id = -1>}, %output: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<ub>, life_time_axis_id = -1>}) {
  %loaded = afir.load %input outs(%output) {
    afir.operand_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<input>>],
    afir.result_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<compute>>],
    afir.destination_descs = [#afir.tensor_desc<axis = [0], role = #afir.role<compute>>],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
  return
}
