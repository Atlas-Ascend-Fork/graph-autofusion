// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details and conditions.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See the License for details and conditions.

// Inductor 的 `AscBackend` 外层图不重复 `Data` 节点的 GM 形状，源 memref 可能退化成 scalar。
// 只有这种源 descriptor 无轴的退化形态才跳过元素数比较；源 descriptor 有轴时仍必须校验。
//
// 反例（没有 scheduler descriptor 的静态 mismatch）仍在 afir_transfer_ops_invalid.mlir。
// 源 descriptor 有轴的 scheduler-tiled transfer 仍必须比较元素数；见文件末尾的反例。

// RUN: af-opt --verify-diagnostics %s

#gm_scalar = #afir.tensor_desc<role = <input>>
#ub_tile = #afir.tensor_desc<axis = [0, 1, 2, 3],
                             repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"5">],
                             strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">, #afir.expr<"1">],
                             role = <compute>>

func.func private @scalar_gm_to_static_ub_tile(
    %input: memref<f32> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"20">,
                                                              life_time_axis_id = -1>},
    %tile: memref<1x1x1x5xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                                  size_from_resource = true,
                                                                  life_time_axis_id = -1>})
    attributes {afir.arg_descs = [#gm_scalar, #ub_tile],
                afir.function_role = #afir.function_role<kernel_body>} {
  %loaded = afir.load %input outs(%tile) {
    afir.operand_descs = [#gm_scalar],
    afir.destination_descs = [#ub_tile],
    afir.result_descs = [#ub_tile],
    afir.sched = #afir.sched<axis = [0, 1, 2, 3], loop_axis = 3>,
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
  } : memref<f32>, memref<1x1x1x5xf32> -> memref<1x1x1x5xf32>
  return
}

// -----

#gm_axis01 = #afir.tensor_desc<axis = [0, 1],
                              repeats = [#afir.expr<"1">, #afir.expr<"4">],
                              strides = [#afir.expr<"0">, #afir.expr<"1">],
                              role = <input>>
#ub_axis23 = #afir.tensor_desc<axis = [2, 3],
                              repeats = [#afir.expr<"1">, #afir.expr<"8">],
                              strides = [#afir.expr<"0">, #afir.expr<"1">],
                              role = <compute>>

// 轴空间不同 + 元素数不同是 scheduler 合轴 / 重切后的边界形态。这里按设计放行，
// 由调度 descriptor 与下降侧消费；不要把该行为收紧成静态元素数校验。
func.func private @scheduler_tiled_axis_space_difference_is_allowed(
    %input: memref<1x4xf32> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"16">,
                                                             life_time_axis_id = -1>},
    %tile: memref<1x8xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                               size_from_resource = true,
                                                               life_time_axis_id = -1>})
    attributes {afir.arg_descs = [#gm_axis01, #ub_axis23],
                afir.function_role = #afir.function_role<kernel_body>} {
  %loaded = afir.load %input outs(%tile) {
    afir.operand_descs = [#gm_axis01],
    afir.destination_descs = [#ub_axis23],
    afir.result_descs = [#ub_axis23],
    afir.sched = #afir.sched<axis = [0, 1, 2, 3], loop_axis = 1>,
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d2, d3)>]
  } : memref<1x4xf32>, memref<1x8xf32> -> memref<1x8xf32>
  return
}

// -----

#gm_tile_4 = #afir.tensor_desc<axis = [0, 1, 2, 3],
                              repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"4">],
                              strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">, #afir.expr<"1">],
                              role = <input>>
#ub_tile_5 = #afir.tensor_desc<axis = [0, 1, 2, 3],
                             repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"5">],
                             strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">, #afir.expr<"1">],
                             role = <compute>>

func.func private @scheduler_tiled_static_element_count_mismatch(
    %input: memref<1x1x1x4xf32> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"16">,
                                                               life_time_axis_id = -1>},
    %tile: memref<1x1x1x5xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                                 size_from_resource = true,
                                                                 life_time_axis_id = -1>})
    attributes {afir.arg_descs = [#gm_tile_4, #ub_tile_5],
                afir.function_role = #afir.function_role<kernel_body>} {
  // expected-error @+1 {{transfer source and destination must have the same static element count}}
  %loaded = afir.load %input outs(%tile) {
    afir.operand_descs = [#gm_tile_4],
    afir.destination_descs = [#ub_tile_5],
    afir.result_descs = [#ub_tile_5],
    afir.sched = #afir.sched<axis = [0, 1, 2, 3], loop_axis = 3>,
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
  } : memref<1x1x1x4xf32>, memref<1x1x1x5xf32> -> memref<1x1x1x5xf32>
  return
}
