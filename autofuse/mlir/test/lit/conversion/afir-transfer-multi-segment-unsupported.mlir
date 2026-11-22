// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 多段 tile 由上游的分段搬运路径处理，能力选择应接受此 Load。

// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation %s | FileCheck %s

#gm_in = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"8">],
                           strides = [#afir.expr<"8">, #afir.expr<"1">], role = <input>>
// 两根向量化轴，外段 stride 16 而内段 extent 8 —— `stride[外] == stride[内] × extent[内]`
// 不成立，合轴封段，这块 tile 剩两段。
#tile = #afir.tensor_desc<axis = [3, 4, 2, 1],
                          repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                     #afir.expr<"zt_size">, #afir.expr<"8">],
                          strides = [#afir.expr<"(16 * zTb_size * zt_size)">, #afir.expr<"(16 * zt_size)">,
                                     #afir.expr<"16">, #afir.expr<"1">],
                          vectorized_axis = [2, 1],
                          vectorized_strides = [#afir.expr<"16">, #afir.expr<"1">],
                          role = <compute>>

func.func private @multi_segment_load(
    %input: memref<128x8xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                               afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4096">,
                                                                    life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm_in],
  afir.asc_graph = #afir.asc_graph<
    tiling_key = 0,
    axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"128">, split_pair_other_id = 0>,
            <id = 1, name = "w", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 1>,
            <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">, from = [0],
             split_pair_other_id = 5>,
            <id = 3, name = "zTB", axis_type = BlockOuter,
             size = #afir.expr<"Ceiling((Ceiling((128 / (zt_size))) / (zTb_size)))">, from = [5],
             split_pair_other_id = 4>,
            <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">, from = [5],
             split_pair_other_id = 3>,
            <id = 5, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((128 / (zt_size)))">,
             from = [0], split_pair_other_id = 2>],
    type = Compute,
    size_var = [#afir.expr<"zt_size">, #afir.expr<"zTb_size">]>,
  afir.function_role = #afir.function_role<kernel_body>
} {
  %tile_size = afir.tiling_param "zt_size" : index
  %block = afir.tiling_param "zTb_size" : index
  %outer = afir.symbolic_eval %block, %tile_size {expr = #afir.expr<"(128 / (__arg0 * __arg1))">}
      : (index, index) -> index
  // 使用动态维保持 GM 与局部侧元素总数均为 128×8，只让 tiling 决定三维布局。
  // 静态不匹配会被 Load 的元素数校验提前拒绝，无法检验多段搬运能力。
  %dst = memref.alloc(%outer, %block, %tile_size) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?x8xf32>
  %loaded = afir.load %input outs(%dst) {
    afir.destination_descs = [#tile],
    afir.operand_descs = [#gm_in],
    afir.result_descs = [#tile],
    afir.sched = #afir.sched<axis = [3, 4, 2, 1], loop_axis = 4>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>]
  } : memref<128x8xf32>, memref<?x?x?x8xf32> -> memref<?x?x?x8xf32>
  return
}

// CHECK: afir.load {{.*}} ascend.implementation = {id = "load.gm_ub", kind = "ascendc_api"}
