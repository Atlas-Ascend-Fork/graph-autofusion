// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 调度路径上计算 op 的**元素个数**：最后一个 tile 取尾块长度，其余取轴长。
//
// 此前没有任何 lit 钉住这条。走 `ScheduledContiguous` 的 lit 只有 Load / Store 与
// `afir.broadcast`（那一档另有 stride 为 0 的特判），计算 op 的 lit 一律停在 identity
// 形态、元素数取自 `local_tensor.get_size`——那条路上压根没有尾块。
//
// 缺了这一条会怎样：`count` 从「带尾块修正的长度」退成「轴长」，最后一趟多算元素。
// 而符号尺寸下 `zt_size` 与总量的整除关系在编译期看不出来，所以这不是某个形状的问题，
// 是所有总量切不齐的图的问题。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// kernel 文本由 `--afir-aggregate-launch` 直接出。
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=elementwise_tail %t.mlir \
// RUN:   | FileCheck %s --check-prefix=CPP
// 走的是调度那一支：元素数不来自 `local_tensor.get_size`（那是 identity 一档，没有尾块）。
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s \
// RUN:   | FileCheck %s --check-prefix=NOIDENTITY

#gm_in = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">], role = <input>>
#gm_out = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">], role = <output>>
#tile = #afir.tensor_desc<axis = [3, 4, 2],
                          repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                     #afir.expr<"zt_size">],
                          strides = [#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">,
                                     #afir.expr<"1">],
                          vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input: memref<128xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                               afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"512">,
                                                                    life_time_axis_id = -1>},
      %output: memref<128xf32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>,
                                afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"512">,
                                                                     life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#gm_in, #gm_out],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"128">, split_pair_other_id = 0>,
              <id = 1, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((128 / (zt_size)))">,
               from = [0], split_pair_other_id = 2>,
              <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">, from = [0],
               split_pair_other_id = 1>,
              <id = 3, name = "zTB", axis_type = BlockOuter,
               size = #afir.expr<"Ceiling((Ceiling((128 / (zt_size))) / (zTb_size)))">, from = [1],
               split_pair_other_id = 4>,
              <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">, from = [1],
               split_pair_other_id = 3>],
      type = Compute,
      size_var = [#afir.expr<"zt_size">, #afir.expr<"zTb_size">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %tmp_bytes = afir.tiling_param "b0_size" : index
    %tile_size = afir.tiling_param "zt_size" : index
    %block = afir.tiling_param "zTb_size" : index
    %outer = afir.symbolic_eval %block, %tile_size {expr = #afir.expr<"(128 / (__arg0 * __arg1))">}
        : (index, index) -> index
    %src = memref.alloc(%outer, %block, %tile_size) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?xf32>
    %loaded = afir.load %input outs(%src) {
      afir.destination_descs = [#tile],
      afir.operand_descs = [#gm_in],
      afir.result_descs = [#tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>]
    } : memref<128xf32>, memref<?x?x?xf32> -> memref<?x?x?xf32>
    %dst = memref.alloc(%outer, %block, %tile_size) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?xf32>
    %tmp = memref.alloc(%tmp_bytes) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                                          size_from_resource = true,
                                                                          alignment_bytes = 32,
                                                                          life_time_axis_id = -1>} : memref<?xui8>
    %computed = afir.abs %loaded outs(%dst) tmp(%tmp) {
      afir.destination_descs = [#tile],
      afir.operand_descs = [#tile],
      afir.result_descs = [#tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>
    } : memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?xui8> -> memref<?x?x?xf32>
    afir.store %computed outs(%output) {
      afir.destination_descs = [#gm_out],
      afir.operand_descs = [#tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d3)>]
    } : memref<?x?x?xf32>, memref<128xf32>
    return
  }
}

// IR-LABEL: func.func private @stage0_template0
// tile 循环只有这一层。计算 op 是单段，多出一层就说明分段或窗口大小算错了。
// IR: scf.for
// IR-NOT: scf.for

// 计算 op 复用循环生成的 actual_size；它在 DeQue 之前已经物化。
// IR: %[[COUNT:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 2, role = actual_size>}
// IR: ascendc.que_bind.deque_tensor
// IR: ascendc.reinterpret_cast
// 本地侧偏移恒为常量 0：单段没有外层循环，偏移里没有归纳变量。
// IR: %[[SRC:.*]] = ascendc.local_tensor.subindex %{{.*}}[%c0
// IR: %[[DST:.*]] = ascendc.local_tensor.subindex %{{.*}}[%c0
// IR: ascendc.abs_extend %[[DST]], %[[SRC]], %{{.*}}, %[[COUNT]]

// kernel 文本侧的同一件事。这一条比 IR 那几条结实：它不关心中间产生了哪些 arith op，
// 只要求**发给算子的那个元素数就是带尾块修正的那个值**。
// 先锚到 tile 循环上：循环之前也有形状相同的三元式（block 维的尾块），不锚会先匹配到那个。
// CPP: for (uint32_t
// CPP: bool [[USETAIL:v[0-9]+]] = {{.*}} < {{.*}};
// CPP: uint32_t [[COUNT:v[0-9]+]] = [[USETAIL]] ? {{.*}} : {{.*}};
// CPP: AbsExtend<float>({{.*}}, {{.*}}, {{.*}}, static_cast<uint32_t>([[COUNT]]));

// NOIDENTITY-NOT: ascendc.local_tensor.get_size
