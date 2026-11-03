// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `exec_condition` 建不出谓词时**报错，不当成 `no_cache` 放过**：放过在生成的 kernel 里看不出
// 任何痕迹，上游那次决策就无声消失了；报错则整份 plan 被拒、调用方回落 legacy。
//
// 另一格「有 `exec_condition` 却没有 `loop_axis`」在这里写不出来：`afir.sched` 的
// assembly format 里 `loop_axis` 排在前面，省掉它之后 `exec_condition` 根本解析不到（试出来
// 的是 `expected 'loop_axis'`）。那条由 `ScheduleAttr` 的 verifier 挡在 C++ 构造路径上，
// 覆盖在 `AfirBuilderTest` 里。

// RUN: not af-opt --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-exec-condition %s 2>&1 | FileCheck %s

// 没有分块切分的调度：轴表里只有一根 ORIGINAL 轴，既没有 BlockOuter 也没有 TileOuter，
// 相位和分核序号都无从谈起。
#gm_in = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">], role = <input>>
#tile = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">],
                          vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>

func.func private @no_split(
    %input: memref<128xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                             afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"512">,
                                                                  life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm_in],
  afir.asc_graph = #afir.asc_graph<
    tiling_key = 0,
    axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"128">, split_pair_other_id = 0>],
    type = Compute>,
  afir.function_role = #afir.function_role<kernel_body>
} {
  %dst = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                                             slot_count = 1, queue_depth = 2,
                                                             size_from_resource = true,
                                                             life_time_axis_id = -1>} : memref<128xf32>
  %loaded = afir.load %input outs(%dst) {
    afir.destination_descs = [#tile],
    afir.operand_descs = [#gm_in],
    afir.result_descs = [#tile],
    afir.sched = #afir.sched<axis = [0], loop_axis = 0, exec_condition = cache_block_split_fused_brc_axis>,
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<128xf32>, memref<128xf32> -> memref<128xf32>
  return
}

// CHECK: UnsupportedOperation: exec_condition cache_block_split_fused_brc_axis needs a block/tile split schedule
