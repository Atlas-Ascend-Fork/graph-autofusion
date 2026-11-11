// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software and is made available under the CANN Open
// Software License Agreement Version 2.0. Nothing in this file may be copied
// or distributed except according to the terms of that license.

// RUN: af-opt --verify-diagnostics --split-input-file %s

// scheduler 删除 redundant Broadcast 后，buffer-form unary 可能收到
// source 为 1×8，destination 为 4×8，不是 tensor-form 的结果类型推导。
// 仅当 descriptor 证明差异轴 stride 为 0，且该轴不参与向量化时才允许这种形状差异。
#src = #afir.tensor_desc<axis = [0, 1],
                         repeats = [#afir.expr<"1">, #afir.expr<"8">],
                         strides = [#afir.expr<"0">, #afir.expr<"1">],
                         vectorized_axis = [1],
                         vectorized_strides = [#afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1],
                         repeats = [#afir.expr<"4">, #afir.expr<"8">],
                         strides = [#afir.expr<"8">, #afir.expr<"1">],
                         vectorized_axis = [1],
                         vectorized_strides = [#afir.expr<"1">], role = <compute>>

func.func @unary_source_broadcasts_to_destination(
    %input: memref<1x8xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>},
    %destination: memref<4x8xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, life_time_axis_id = -1>})
      attributes {afir.arg_descs = [#afir.tensor_desc<role = <input>>, #afir.tensor_desc<role = <output>>]} {
  %result = afir.exp %input outs(%destination) {
    afir.operand_descs = [#src],
    afir.destination_descs = [#dst],
    afir.result_descs = [#dst]
  } : memref<1x8xf32>, memref<4x8xf32>
      -> memref<4x8xf32>
  return
}

// -----

#dst = #afir.tensor_desc<axis = [0, 1],
                         repeats = [#afir.expr<"4">, #afir.expr<"8">],
                         strides = [#afir.expr<"8">, #afir.expr<"1">],
                         vectorized_axis = [1],
                         vectorized_strides = [#afir.expr<"1">], role = <compute>>

// 广播轴的 stride 不是 0：这不是重复读同一块 source，而是两张真正不同的视图。
#src_nonzero_stride = #afir.tensor_desc<axis = [0, 1],
                                        repeats = [#afir.expr<"1">, #afir.expr<"8">],
                                        strides = [#afir.expr<"1">, #afir.expr<"1">],
                                        vectorized_axis = [1],
                                        vectorized_strides = [#afir.expr<"1">], role = <compute>>

func.func @unary_nonzero_broadcast_stride_is_invalid(
    %input: memref<1x8xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>},
    %destination: memref<4x8xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, life_time_axis_id = -1>})
      attributes {afir.arg_descs = [#afir.tensor_desc<role = <input>>, #afir.tensor_desc<role = <output>>]} {
  // expected-error @+1 {{broadcast source stride must be zero}}
  %result = afir.exp %input outs(%destination) {
    afir.operand_descs = [#src_nonzero_stride],
    afir.destination_descs = [#dst],
    afir.result_descs = [#dst]
  } : memref<1x8xf32>, memref<4x8xf32>
      -> memref<4x8xf32>
  return
}

// -----

#dst = #afir.tensor_desc<axis = [0, 1],
                         repeats = [#afir.expr<"4">, #afir.expr<"8">],
                         strides = [#afir.expr<"8">, #afir.expr<"1">],
                         vectorized_axis = [1],
                         vectorized_strides = [#afir.expr<"1">], role = <compute>>

// 向量化轴上不能广播：AscendC API 按目的地窗口取 count，会把 1 元素窗口读成 8 元素。
#src_vectorized_broadcast = #afir.tensor_desc<axis = [0, 1],
                                              repeats = [#afir.expr<"1">, #afir.expr<"8">],
                                              strides = [#afir.expr<"0">, #afir.expr<"1">],
                                              vectorized_axis = [0, 1],
                                              vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">],
                                              role = <compute>>

func.func @unary_vectorized_broadcast_is_invalid(
    %input: memref<1x8xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>},
    %destination: memref<4x8xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, life_time_axis_id = -1>})
      attributes {afir.arg_descs = [#afir.tensor_desc<role = <input>>, #afir.tensor_desc<role = <output>>]} {
  // expected-error @+1 {{broadcast axis must not be vectorized}}
  %result = afir.exp %input outs(%destination) {
    afir.operand_descs = [#src_vectorized_broadcast],
    afir.destination_descs = [#dst],
    afir.result_descs = [#dst]
  } : memref<1x8xf32>, memref<4x8xf32>
      -> memref<4x8xf32>
  return
}

// -----

#two_input_source = #afir.tensor_desc<axis = [0, 1],
                                      repeats = [#afir.expr<"8">, #afir.expr<"1">],
                                      strides = [#afir.expr<"1">, #afir.expr<"0">],
                                      role = <compute>>
#two_input_destination = #afir.tensor_desc<axis = [0, 1],
                                           repeats = [#afir.expr<"8">, #afir.expr<"8">],
                                           strides = [#afir.expr<"8">, #afir.expr<"1">],
                                           role = <compute>>

func.func private @binary_operands_both_broadcast_to_destination(
    %lhs: memref<8x1xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                              life_time_axis_id = -1>},
    %rhs: memref<8x1xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1,
                                                               life_time_axis_id = -1>},
    %destination: memref<8x8xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2,
                                                                       life_time_axis_id = -1>})
    attributes {afir.arg_descs = [#afir.tensor_desc<role = <input>>,
                                  #afir.tensor_desc<role = <input>>,
                                  #afir.tensor_desc<role = <output>>],
                afir.function_role = #afir.function_role<kernel_body>} {
  %result = afir.add %lhs, %rhs outs(%destination) {
    afir.operand_descs = [#two_input_source, #two_input_source],
    afir.destination_descs = [#two_input_destination],
    afir.result_descs = [#two_input_destination]
  } : (memref<8x1xf32>, memref<8x1xf32>), memref<8x8xf32> -> memref<8x8xf32>
  return
}

// -----

#wrong_outer_pitch_source = #afir.tensor_desc<axis = [0, 1],
                                              repeats = [#afir.expr<"8">, #afir.expr<"1">],
                                              strides = [#afir.expr<"1">, #afir.expr<"0">],
                                              role = <compute>>
#wrong_outer_pitch_destination = #afir.tensor_desc<axis = [0, 1],
                                                   repeats = [#afir.expr<"8">, #afir.expr<"8">],
                                                   strides = [#afir.expr<"9">, #afir.expr<"1">],
                                                   role = <compute>>

func.func private @broadcast_outer_pitch_must_match_inner_extent(
    %lhs: memref<8x1xf32>,
    %destination: memref<8x8xf32>) {
  // expected-error @+1 {{destination outer stride must equal source stride multiplied by later broadcast extents}}
  %result = afir.exp %lhs outs(%destination) {
    afir.operand_descs = [#wrong_outer_pitch_source],
    afir.destination_descs = [#wrong_outer_pitch_destination],
    afir.result_descs = [#wrong_outer_pitch_destination]
  } : memref<8x1xf32>, memref<8x8xf32> -> memref<8x8xf32>
  return
}
