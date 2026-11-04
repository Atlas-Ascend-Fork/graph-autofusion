// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s --implicit-check-not=emitasc.member
// RUN: not af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate --afir-memory-allocate %s 2>&1 | FileCheck %s --check-prefix=REPEAT

module {
  func.func private @buffer_form_copy(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %output: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], role = <input>>,
      #afir.tensor_desc<axis = [0], role = <output>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %input outs(%tmp) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    afir.store %loaded outs(%output) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// CHECK-LABEL: func.func private @buffer_form_copy
// CHECK-SAME: (%{{[A-Za-z0-9_]+}}: memref<16xf16, 22 : i32>
// CHECK-SAME: %{{[A-Za-z0-9_]+}}: memref<16xf16, 22 : i32>
// CHECK-SAME: memref<ui8, 22 : i32>
// CHECK-SAME: !emitasc.py_struct<"TilingData", [i64, i64, i64], ["block_dim", "s0", "q2_size"]>
// 运行期尺寸在 AFIR 层就是一个 SSA 值。此前这里是两个 `emitasc.member` 加两次
// index cast——循环生成与资源物化各取一遍 `s0`，而 `emitasc` 是后端方言，出现在这个
// 阶段的函数体里就让判据 C3 当场不成立。现在两处共用同一个 `afir.tiling_param`，
// `emitasc.member` 只在 lowering 里出现一次（RUN 行的 implicit-check-not 钉住这条）。
// CHECK: %[[S0:.*]] = afir.tiling_param "s0"
// 「资源已物化」不再由 `afir.ascendc_materialized` 属性宣告——下面这行 `ascendc.pipe`
// 就是它的结构证据，判据见 conversion/AFIRToAscendC/AFIRToAscendCLoweringStage.h。
// CHECK: %[[PIPE:.*]] = ascendc.pipe
// queue_depth 由 QueueType 的第二个参数承载，不再另发扁平属性（设计文档 I2/I3，5.1 步 2）。
// CHECK: %[[QUEUE:.*]] = ascendc.queue : <vecin, 2>
// CHECK: arith.ceildivsi
// CHECK: arith.mulsi_extended
// CHECK: cf.assert
// CHECK: ascendc.pipe.init_queue %[[PIPE]], %[[QUEUE]]
// CHECK: ascendc.global_tensor
// CHECK: ascendc.global_tensor.set_global_buffer %{{.*}}, %arg0
// CHECK: ascendc.global_tensor.set_global_buffer %{{.*}}, %arg1
// 循环上界用的就是上面那个值，没有第二次取。
// CHECK: scf.for %{{.*}} to %[[S0]]
// CHECK: ascendc.que_bind.alloc_tensor %[[QUEUE]]
// CHECK: ascendc.reinterpret_cast
// CHECK: afir.load
// CHECK: afir.store
// CHECK: ascendc.que_bind.free_tensor %[[QUEUE]]
// CHECK-NOT: memref.alloc
// REPEAT: canonical AFIR resources are already materialized
