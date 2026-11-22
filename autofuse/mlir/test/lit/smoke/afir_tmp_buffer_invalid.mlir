// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --verify-diagnostics --split-input-file %s

// 操作数分 inputs | destinations | tmpBuffers 三段。把 tmp buffer 当成一个输入是这一类最
// 容易犯的错，而且它的表现是静默错位而不是崩溃：descriptor 少一条、tie 指向
// tmp buffer、竞争检查把 destination 当成读。所以每一段的段长都要有一条负例钉住。

#compute = #afir.tensor_desc<axis = [0], role = <compute>>
#tile = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2,>
#tmp_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 7, size_from_resource = true, alignment_bytes = 32,>

// 按旧规则把 tmp buffer 数成输入：三条 operand_descs。
func.func @tmp_buffer_counted_as_input(
    %lhs: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile},
    %rhs: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile},
    %out: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile}) {
  %tmp = memref.alloc() {afir.buffer_desc = #tmp_buffer} : memref<8192xui8, #afir.space<ub>>
  // expected-error @+1 {{DescriptorCountMismatch: 'afir.operand_descs' expected 2 element but found 3}}
  %sum = afir.add %lhs, %rhs outs(%out) tmp(%tmp) {
    afir.destination_descs = [#compute],
    afir.operand_descs = [#compute, #compute, #compute],
    afir.result_descs = [#compute]
  } : (memref<16xf16, #afir.space<ub>>, memref<16xf16, #afir.space<ub>>),
      memref<16xf16, #afir.space<ub>>, memref<8192xui8, #afir.space<ub>> -> memref<16xf16, #afir.space<ub>>
  return
}

// -----

#compute = #afir.tensor_desc<axis = [0], role = <compute>>
#tile = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2,>
#tmp_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 7, size_from_resource = true, alignment_bytes = 32,>

// 反过来：tmp buffer 也不是 destination，destination_descs 只能有一条。
func.func @tmp_buffer_counted_as_destination(
    %input: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile},
    %out: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile}) {
  %tmp = memref.alloc() {afir.buffer_desc = #tmp_buffer} : memref<8192xui8, #afir.space<ub>>
  // expected-error @+1 {{DescriptorCountMismatch: 'afir.destination_descs' expected 1 element but found 2}}
  %abs = afir.abs %input outs(%out) tmp(%tmp) {
    afir.destination_descs = [#compute, #compute],
    afir.operand_descs = [#compute],
    afir.result_descs = [#compute]
  } : memref<16xf16, #afir.space<ub>>, memref<16xf16, #afir.space<ub>>,
      memref<8192xui8, #afir.space<ub>> -> memref<16xf16, #afir.space<ub>>
  return
}

// -----

// tmp buffer 是一块字节，不是某个 dtype 的张量。
func.func @tmp_buffer_is_not_a_typed_tile(%input: memref<16xf16>, %out: memref<16xf16>,
                                       %tmp: memref<8192xf16>) {
  // expected-error @+1 {{operand #2 must be}}
  %abs = afir.abs %input outs(%out) tmp(%tmp) : memref<16xf16>, memref<16xf16>,
      memref<8192xf16> -> memref<16xf16>
  return
}

// -----

// 元素类型在输入与结果之间必须一致。ODS 的类型列表是并集，光靠它 `f16 -> f32` 也
// 能过，而没有哪个 elementwise Op 是这个意思。
func.func @element_type_must_not_change(%input: tensor<16xf16>) {
  // expected-error @+1 {{all of {input, result} have same element type}}
  %abs = afir.abs %input : tensor<16xf16> -> tensor<16xf32>
  return
}

// -----

// 二元同理，而且 buffer 形态下 destination 还要与结果同型（那条判据在 verifier 里）。
func.func @binary_element_type_must_agree(%lhs: tensor<16xf16>, %rhs: tensor<16xf32>) {
  // expected-error @+1 {{all of {lhs, rhs, result} have same element type}}
  %sum = afir.add %lhs, %rhs : (tensor<16xf16>, tensor<16xf32>) -> tensor<16xf16>
  return
}
