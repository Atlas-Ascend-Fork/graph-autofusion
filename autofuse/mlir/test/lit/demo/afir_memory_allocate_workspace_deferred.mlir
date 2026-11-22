// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of
// the CANN Open Software License Agreement Version 2.0.
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s

module {
  // 以 ascendc.pipe 等 AscendC 算子作为资源已物化的结构证据，
  // 不依赖 memory_allocated 阶段标记，判据见 LoweringStage.h。
  func.func private @workspace_plan(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"16">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 99, size_bytes_expr = <"256">, life_time_axis_id = -1>],
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2,
        size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %input outs(%tmp) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    return
  }
}

// CHECK-LABEL: func.func private @workspace_plan
// CHECK: ascendc.pipe
