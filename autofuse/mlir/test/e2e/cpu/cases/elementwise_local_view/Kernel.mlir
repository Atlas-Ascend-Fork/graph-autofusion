// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#dense = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"4">, <"8">],
  strides = [<"gm_outer">, <"gm_row">, <"gm_col">],
  vectorized_axis = [1, 2], vectorized_strides = [<"8">, <"1">], role = <compute>>
#pitched = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"4">, <"8">],
  strides = [<"gm_outer">, <"gm_row">, <"gm_col">],
  vectorized_axis = [1, 2], vectorized_strides = [<"16">, <"1">], role = <compute>>

#src = #afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"16">, <"0">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"8">, <"0">], role = <compute>>

#span_src = #afir.tensor_desc<axis = [0, 1], repeats = [<"2">, <"1">], strides = [<"1">, <"0">],
  vectorized_axis = [0, 1], vectorized_strides = [<"8">, <"0">], role = <compute>>
#span_dst = #afir.tensor_desc<axis = [0, 1], repeats = [<"2">, <"1">], strides = [<"1">, <"0">],
  vectorized_axis = [0, 1], vectorized_strides = [<"8">, <"0">], role = <compute>>

#src_repeat_tail = #afir.tensor_desc<axis = [0, 1], repeats = [<"257">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"16">, <"0">], role = <compute>>
#dst_repeat_tail = #afir.tensor_desc<axis = [0, 1], repeats = [<"257">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"8">, <"0">], role = <compute>>

#src_wide_stride = #afir.tensor_desc<axis = [0, 1], repeats = [<"2">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"4096">, <"0">], role = <compute>>
#dst_wide_stride = #afir.tensor_desc<axis = [0, 1], repeats = [<"2">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"2048">, <"0">], role = <compute>>

#src_equal_stride = #afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"16">, <"0">], role = <compute>>
#dst_equal_stride = #afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"16">, <"0">], role = <compute>>

module {
  func.func @neg_span(%src: !ascendc.local_tensor<16xf32>, %dst: !ascendc.local_tensor<16xf32>) {
    %r = afir.neg %src outs(%dst) {afir.operand_descs = [#span_src], afir.destination_descs = [#span_dst], afir.result_descs = [#span_dst]}
      : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32> -> !ascendc.local_tensor<16xf32>
    return
  }
  func.func @muls_span(%src: !ascendc.local_tensor<16xf32>, %scalar: f32, %dst: !ascendc.local_tensor<16xf32>) {
    %r = afir.mul %src, %scalar outs(%dst) {afir.operand_descs = [#span_src, #afir.tensor_desc<role = <compute>>], afir.destination_descs = [#span_dst], afir.result_descs = [#span_dst]}
      : (!ascendc.local_tensor<16xf32>, f32), !ascendc.local_tensor<16xf32> -> !ascendc.local_tensor<16xf32>
    return
  }
  func.func @mul_span(%src: !ascendc.local_tensor<16xf32>, %rhs: !ascendc.local_tensor<16xf32>, %dst: !ascendc.local_tensor<16xf32>) {
    %r = afir.mul %src, %rhs outs(%dst) {afir.operand_descs = [#span_src, #span_src], afir.destination_descs = [#span_dst], afir.result_descs = [#span_dst]}
      : (!ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>), !ascendc.local_tensor<16xf32> -> !ascendc.local_tensor<16xf32>
    return
  }

  func.func @abs_different_ub(%a: !ascendc.local_tensor<2x4x8xf32>, %dst: !ascendc.local_tensor<2x4x8xf32>, %tmp: !ascendc.local_tensor<128xui8>) {
    %r = afir.abs %a outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#dense], afir.destination_descs = [#pitched], afir.result_descs = [#pitched]
    } : !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<128xui8> -> !ascendc.local_tensor<2x4x8xf32>
    return
  }
  func.func @add_different_ub(%a: !ascendc.local_tensor<2x4x8xf32>, %b: !ascendc.local_tensor<2x4x8xf32>, %dst: !ascendc.local_tensor<2x4x8xf32>) {
    %r = afir.add %a, %b outs(%dst) {
      afir.operand_descs = [#dense, #pitched], afir.destination_descs = [#pitched], afir.result_descs = [#pitched]
    } : (!ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>), !ascendc.local_tensor<2x4x8xf32> -> !ascendc.local_tensor<2x4x8xf32>
    return
  }
  func.func @clip_different_ub(%a: !ascendc.local_tensor<2x4x8xf32>, %b: !ascendc.local_tensor<2x4x8xf32>, %c: !ascendc.local_tensor<2x4x8xf32>, %dst: !ascendc.local_tensor<2x4x8xf32>, %tmp: !ascendc.local_tensor<128xui8>) {
    %r = afir.clip_by_value %a, %b, %c outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#dense, #pitched, #dense], afir.destination_descs = [#pitched], afir.result_descs = [#pitched]
    } : (!ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>), !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<128xui8> -> !ascendc.local_tensor<2x4x8xf32>
    return
  }

  func.func @cast_padded_f16(%input: !ascendc.local_tensor<?xf16>,
      %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.cast %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
  func.func @cast_padded_bf16(%input: !ascendc.local_tensor<?xbf16>,
      %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.cast %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst]
    } : !ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
  func.func @cast_repeat_tail(%input: !ascendc.local_tensor<?xf16>,
      %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.cast %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#src_repeat_tail], afir.destination_descs = [#dst_repeat_tail], afir.result_descs = [#dst_repeat_tail]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
  func.func @cast_wide_stride(%input: !ascendc.local_tensor<?xf16>,
      %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.cast %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#src_wide_stride], afir.destination_descs = [#dst_wide_stride], afir.result_descs = [#dst_wide_stride]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
  func.func @cast_equal_stride(%input: !ascendc.local_tensor<?xf16>,
      %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.cast %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#src_equal_stride], afir.destination_descs = [#dst_equal_stride], afir.result_descs = [#dst_equal_stride]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}
