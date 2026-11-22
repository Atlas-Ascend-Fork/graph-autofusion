// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// RUN: af-opt --verify-diagnostics %s

// The bridge proves this relation before constructing the operation.  The
// verifier rebuilds it from the canonical descriptors below, so an externally
// injected proof attribute cannot override the descriptor facts.

func.func @dynamic_transfer_shape_mismatch(%input: memref<?xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>}, %output: memref<?xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>})
    attributes {
      afir.arg_descs = [
        #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"s0">], strides = [#afir.expr<"1">], role = <input>>,
        #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"s1">], strides = [#afir.expr<"1">], role = <output>>
      ]
    } {
  // expected-error @+1 {{UnsupportedOperation: unknown AFIR operation attribute '"afir.transfer_element_count_proof"'}}
  %loaded = afir.load %input outs(%output) {
    // This legacy-looking attribute is intentionally ignored by the verifier.
    afir.transfer_element_count_proof = "True",
    afir.operand_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"s0">], strides = [#afir.expr<"1">], role = <input>>],
    afir.destination_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"s1">], strides = [#afir.expr<"1">], role = <compute>>],
    afir.result_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"s1">], strides = [#afir.expr<"1">], role = <compute>>],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<?xf16>, memref<?xf16> -> memref<?xf16>
  return
}
