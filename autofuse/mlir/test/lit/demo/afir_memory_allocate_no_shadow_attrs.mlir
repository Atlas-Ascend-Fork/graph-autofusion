// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// AFIRMemoryAllocate 不再写出影子属性。
//
// MemoryAllocate 曾把资源身份与规格以扁平属性的形式回写到生成的 AscendC Op 上
// （afir.que_id / afir.buf_id / afir.buffer_id / afir.queue_depth / afir.slot_count /
// afir.alignment_bytes / afir.reuse_id / afir.life_time_axis_id）。它们在 MLIR 树内
// 只有写点、没有读点，且每一条都是同一事实的第二份编码——设计文档 I2 / I3 / D4，
// 5.1 步 2 予以删除。
//
// 本用例覆盖两条资源路径（isQueueBinding 为真走 TQue、为假走 TBuf），并同时断言
// 两件事：
//   1) NOSHADOW 前缀——上述扁平属性一个都不再出现；
//   1b) NOSTAGE 前缀——四个记录「哪个 pass 跑过」的阶段标记属性也不再出现
//       （设计文档 D5 / 验收 8）。它们同样是第二份编码：每个阶段的产物就摆在 IR 里，
//       判据见 dialect/AFIR/Transforms/LoopGenerate.h 与
//       conversion/AFIRToAscendC/AFIRToAscendCLoweringStage.h；
//   2) CHECK 部分——被它们复述的事实仍然结构性地存在于 IR 里，即删除是去重而非丢信息。
//
// 事实的唯一落点对照：
//   资源身份      ascendc.tbuf / ascendc.queue 的 SSA 值本身
//   queue_depth   QueueType 的第二个参数，如 <vecin, 2>
//   slot_count    ascendc.pipe.init_queue 的 count 操作数
//   alignment     已折进 size 计算（ceildivsi/mulsi_extended + 溢出断言）
//   buffer_id /
//   reuse_id /
//   life_time_axis_id
//                 afir Op 上强类型的 #afir.buffer_desc，MemoryAllocate 的输入本身

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s --check-prefix=NOSHADOW
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s --check-prefix=NOSTAGE

module {
  func.func private @tbuf_path(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %output: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>, #afir.tensor_desc<axis = [0], role = <output>>],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute, size_var = [#afir.expr<"s0">]>,
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
    %scratch = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 5, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %sum = afir.add %loaded, %loaded outs(%scratch) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>, #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    %staged = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 9, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %scaled = afir.add %sum, %sum outs(%staged) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>, #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %scaled outs(%output) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// 一个资源一个 TilingData 字段。`%tmp`（buffer 2）既是 `afir.load` 的落点、又是
// `afir.add` 的操作数，但队列身份只看 descriptor 的 `slot_count` / `queue_depth`
// （`isCanonicalQueueResource`），与它在哪些使用点出现无关，所以它只出 `q2_size`，
// 不会同时出 `b2_size`——后者会让 host 侧为同一块存储填两个字段，且与
// `AFIRMemoryAllocate` 的 `isQueueBinding` 判定对不上。`%scratch`（buffer 5）没有
// slot_count / queue_depth，出 `b5_size`。
// CHECK-LABEL: func.func private @tbuf_path
// CHECK-SAME: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["block_dim", "s0", "q2_size", "q9_size", "b5_size"]> {afir.kernel_arg = #afir.kernel_arg<role = tiling>}

// TBuf 路径（isQueueBinding 为假，buffer id 5）：资源就是这个 SSA 值，不带任何属性。
// alignment_bytes = 32 没有被复述成属性，而是折进了 size 计算。
// CHECK: %[[TBUF:.*]] = ascendc.tbuf : <veccalc>
// CHECK: arith.ceildivsi %{{.*}}, %c32
// CHECK: %[[TBUF_SIZE:.*]], %{{.*}} = arith.mulsi_extended
// CHECK: cf.assert %{{.*}}, "aligned resource size must not overflow"
// CHECK: ascendc.pipe.init_buffer %{{.*}}, %[[TBUF]], %[[TBUF_SIZE]]

// TQue 路径（isQueueBinding 为真，buffer id 2 / 9）：queue_depth 在 QueueType 里，
// slot_count 是 init_queue 的 count 操作数。
// CHECK: %[[QIN:.*]] = ascendc.queue : <vecin, 2>
// CHECK: ascendc.pipe.init_queue %{{.*}}, %[[QIN]], %c2_i32
// CHECK: %[[QOUT:.*]] = ascendc.queue : <vecout, 2>
// CHECK: ascendc.pipe.init_queue %{{.*}}, %[[QOUT]], %c2_i32

// 取用点同样不带影子属性；身份由 SSA use-def 给出。
// CHECK: ascendc.tbuf.get_tensor %[[TBUF]]
// 物化之后规划结果跟着新的定义点走，仍然一个值一份（设计文档 5.1 步 8）。
// CHECK: ascendc.reinterpret_cast %{{.*}} {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 5, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>}
// CHECK: ascendc.que_bind.alloc_tensor %[[QIN]]
// CHECK: ascendc.que_bind.alloc_tensor %[[QOUT]]

// buffer_id / slot_count / queue_depth / alignment_bytes / life_time_axis_id 仍在强类型
// descriptor 里，删除影子属性之后它是这些事实的唯一编码；步 8 起这份 descriptor
// 挂在值的定义点上（输入侧的 `memref.alloc`，物化后的 AscendC 定义 Op），Op 上不再
// 有任何按下标对齐的副本。
// CHECK: afir.add
// CHECK-NOT: afir.operand_buffer_descs
// CHECK-NOT: afir.result_buffer_descs
// CHECK-NOT: afir.destination_buffer_descs
// CHECK-NOT: afir.arg_buffer_descs

// NOSHADOW-NOT: afir.que_id
// NOSHADOW-NOT: afir.buf_id
// NOSHADOW-NOT: afir.buffer_id
// NOSHADOW-NOT: afir.queue_depth
// NOSHADOW-NOT: afir.slot_count
// NOSHADOW-NOT: afir.alignment_bytes
// NOSHADOW-NOT: afir.reuse_id
// NOSHADOW-NOT: afir.life_time_axis_id
// NOSHADOW-NOT: afir.axis_desc

// 阶段标记属性随验收 8 删除，IR 里不应再出现任何一个。
// NOSTAGE-NOT: afir.loop_generated
// NOSTAGE-NOT: afir.memory_allocated
// NOSTAGE-NOT: afir.ascendc_materialized
// NOSTAGE-NOT: afir.pipeline_synchronized
