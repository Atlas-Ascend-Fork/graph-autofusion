// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt %s | af-opt | FileCheck %s

// Bufferizable ops accept optional scratch for AscendC APIs that require it.
// size_from_resource refers to ATT's aggregated b{id}_size; AFIR does not compute
// that allocation size (see the elementwise slice design's tmp buffer section).
//
// Scratch is a ui8 memref of bytes, not a tensor value of the computation, so it has
// no tensor descriptor. getNumTmpBuffers() excludes it from descriptor counts.

#compute = #afir.tensor_desc<axis = [0], role = <compute>>
#gm = #afir.buffer_desc<space = <gm>,>
#tile = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, life_time_axis_id = 1>
#tmp_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 7, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = 1>

#graph = #afir.asc_graph<tiling_key = 0,
  axes = [<id = 0, name = "m", axis_type = Original, size = #afir.expr<"s0">>,
          <id = 1, name = "n", axis_type = TileInner, size = #afir.expr<"16">>],
  type = Compute, size_var = [#afir.expr<"s0">]>

// CHECK-LABEL: func.func private @add_with_tmp_buffer
func.func private @add_with_tmp_buffer(
    %lhs: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile},
    %rhs: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile},
    %out: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile}) attributes {
  afir.function_role = #afir.function_role<kernel_body>,
  afir.arg_descs = [#compute, #compute, #compute],
  afir.asc_graph = #graph
} {
  // CHECK: %[[TMP:.*]] = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 7, size_from_resource = true
  %tmp = memref.alloc() {afir.buffer_desc = #tmp_buffer} : memref<8192xui8, #afir.space<ub>>
  // CHECK: afir.add %{{.*}}, %{{.*}} outs(%{{.*}}) tmp(%[[TMP]])
  // CHECK-SAME: -> memref<16xf16, #afir.space<ub>>
  %sum = afir.add %lhs, %rhs outs(%out) tmp(%tmp) {
    afir.destination_descs = [#compute],
    afir.operand_descs = [#compute, #compute],
    afir.result_descs = [#compute],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (memref<16xf16, #afir.space<ub>>, memref<16xf16, #afir.space<ub>>),
      memref<16xf16, #afir.space<ub>>, memref<8192xui8, #afir.space<ub>> -> memref<16xf16, #afir.space<ub>>
  return
}

// The operand stays optional: today's buffer-form operations omit it and print
// exactly as they did before.
// CHECK-LABEL: func.func private @add_without_tmp_buffer
func.func private @add_without_tmp_buffer(
    %lhs: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile},
    %rhs: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile},
    %out: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile}) attributes {
  afir.function_role = #afir.function_role<kernel_body>,
  afir.arg_descs = [#compute, #compute, #compute],
  afir.asc_graph = #graph
} {
  // CHECK: afir.add %{{.*}}, %{{.*}} outs(%{{.*}}) {
  // CHECK-NOT: tmp(
  %sum = afir.add %lhs, %rhs outs(%out) {
    afir.destination_descs = [#compute],
    afir.operand_descs = [#compute, #compute],
    afir.result_descs = [#compute],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (memref<16xf16, #afir.space<ub>>, memref<16xf16, #afir.space<ub>>),
      memref<16xf16, #afir.space<ub>> -> memref<16xf16, #afir.space<ub>>
  return
}
