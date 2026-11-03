// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of
// the CANN Open Software License Version 2.0.
// RUN: af-opt %s | FileCheck %s

module {
  func.func private @dynamic_transfer(%input: memref<?xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>}, %output: memref<?xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"s0">], strides = [#afir.expr<"1">], role = <input>>,
      #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"s0">], strides = [#afir.expr<"1">], role = <output>>
    ],
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %loaded = afir.load %input outs(%output) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"s0">], strides = [#afir.expr<"1">], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"s0">], strides = [#afir.expr<"1">], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"s0">], strides = [#afir.expr<"1">], role = <compute>>],
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<?xf16>, memref<?xf16> -> memref<?xf16>
    return
  }
}

// CHECK-NOT: transfer_element_count_proof
