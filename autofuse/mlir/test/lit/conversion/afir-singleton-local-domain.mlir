// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s
// Only the outer, non-vectorized axis broadcasts. Each local invocation
// handles one element and must use the tensor implementation unambiguously.
#one = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"1">], strides = [#afir.expr<"0">, #afir.expr<"0">], vectorized_axis = [1], vectorized_strides = [#afir.expr<"0">], role = <compute>>
#full = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"ks0">, #afir.expr<"1">], strides = [#afir.expr<"1">, #afir.expr<"0">], vectorized_axis = [1], vectorized_strides = [#afir.expr<"0">], role = <compute>>

func.func @add(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.add %input, %one outs(%dst) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xf32> -> !ascendc.local_tensor<?x1xf32>
  return
}
// CHECK-LABEL: func.func @add
// CHECK: id = "add.tensor_tensor
// CHECK: return

func.func @mul(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.mul %input, %one outs(%dst) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xf32> -> !ascendc.local_tensor<?x1xf32>
  return
}
// CHECK-LABEL: func.func @mul
// CHECK: id = "mul.tensor_tensor
// CHECK: return

func.func @minimum(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.minimum %input, %one outs(%dst) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xf32> -> !ascendc.local_tensor<?x1xf32>
  return
}
// CHECK-LABEL: func.func @minimum
// CHECK: id = "minimum.tensor_tensor
// CHECK: return

func.func @maximum(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.maximum %input, %one outs(%dst) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xf32> -> !ascendc.local_tensor<?x1xf32>
  return
}
// CHECK-LABEL: func.func @maximum
// CHECK: id = "maximum.tensor_tensor
// CHECK: return

func.func @sub(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.sub %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xf32>
  return
}
// CHECK-LABEL: func.func @sub
// CHECK: id = "sub.tensor_tensor
// CHECK: return

func.func @div(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.div %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xf32>
  return
}
// CHECK-LABEL: func.func @div
// CHECK: id = "div.tensor_tensor
// CHECK: return

func.func @truediv(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.truediv %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xf32>
  return
}
// CHECK-LABEL: func.func @truediv
// CHECK: id = "truediv.tensor_tensor
// CHECK: return

func.func @pow(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.pow %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xf32>
  return
}
// CHECK-LABEL: func.func @pow
// CHECK: id = "pow.tensor_tensor
// CHECK: return

func.func @where(%cond: !ascendc.local_tensor<?x1xui8>, %input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.where %cond, %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>,
    afir.operand_descs = [#full, #full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xui8>, !ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xf32>
  return
}
// CHECK-LABEL: func.func @where
// CHECK: id = "where.tensor
// CHECK: return

func.func @select(%cond: !ascendc.local_tensor<?x1xui8>, %input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.select %cond, %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>,
    afir.operand_descs = [#full, #full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xui8>, !ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xf32>
  return
}
// CHECK-LABEL: func.func @select
// CHECK: id = "select.tensor
// CHECK: return

func.func @clip_by_value(%cond: !ascendc.local_tensor<?x1xf32>, %input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.clip_by_value %cond, %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>,
    afir.operand_descs = [#full, #full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xf32>
  return
}
// CHECK-LABEL: func.func @clip_by_value
// CHECK: id = "clip_by_value.tensor
// CHECK: return

func.func @logical_and(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xui8>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.logical_and %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xui8>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xui8>
  return
}
// CHECK-LABEL: func.func @logical_and
// CHECK: id = "logical_and.tensor
// CHECK: return

func.func @logical_or(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xui8>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.logical_or %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xui8>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xui8>
  return
}
// CHECK-LABEL: func.func @logical_or
// CHECK: id = "logical_or.tensor
// CHECK: return

func.func @eq(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xui8>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.eq %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xui8>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xui8>
  return
}
// CHECK-LABEL: func.func @eq
// CHECK: id = "eq.tensor
// CHECK: return

func.func @ne(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xui8>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.ne %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xui8>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xui8>
  return
}
// CHECK-LABEL: func.func @ne
// CHECK: id = "ne.tensor
// CHECK: return

func.func @lt(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xui8>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.lt %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xui8>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xui8>
  return
}
// CHECK-LABEL: func.func @lt
// CHECK: id = "lt.tensor
// CHECK: return

func.func @le(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xui8>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.le %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xui8>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xui8>
  return
}
// CHECK-LABEL: func.func @le
// CHECK: id = "le.tensor
// CHECK: return

func.func @gt(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xui8>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.gt %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xui8>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xui8>
  return
}
// CHECK-LABEL: func.func @gt
// CHECK: id = "gt.tensor
// CHECK: return

func.func @ge(%input: !ascendc.local_tensor<?x1xf32>, %dst: !ascendc.local_tensor<?x1xui8>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1x1xf32>
  %r = afir.ge %input, %one outs(%dst) tmp(%aux) {
    afir.sched = #afir.sched<axis = [2, 3, 1], loop_axis = 3, exec_condition = cache_block_split_fused_brc_axis>, afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<?x1xf32>, !ascendc.local_tensor<1x1xf32>), !ascendc.local_tensor<?x1xui8>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x1xui8>
  return
}
// CHECK-LABEL: func.func @ge
// CHECK: id = "ge.tensor
// CHECK: return
