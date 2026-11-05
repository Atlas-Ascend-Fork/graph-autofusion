// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend950DT_950x %s -o %t.regbase.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=fused_graph %t.regbase.mlir | FileCheck %s --check-prefix=REGBASE
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.membased.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=fused_graph %t.membased.mlir | FileCheck %s --check-prefix=MEMBASED

// Autofuse API extension headers reach the kernel through three separate
// layers, and this pins the first two of them. Operations declare which header
// declares their API (ApiExtensionOpInterface); AddKernelIncludes turns the
// declarations into `emitc.include` operations. The third layer -- replacing
// the include line with the header text -- runs only in the production codegen
// path, so what a LIT run sees is the readable include form.
//
// 两个 target 都选 `load.gm_ub`，都发 `DataCopyPadExtend`、都要 `datacopy.h`。
// **这一点从前不是这样**：mem-based 一侧曾按 `TargetFeature::DataCopyPadExtend` 回落到裸
// `AscendC::DataCopy`、不要任何扩展头，而那个包装其实定义在 mem 侧的
// `autofuse/ascendc/api/datacopy.h`，legacy 两侧都无条件发它。裸 `DataCopy` 要求
// `count * sizeof(T)` 32B 对齐，尾块不对齐时 NPU 上不报错、按 32B 向下取整。
//
// 所以两条 RUN 今天钉的不是「同一个计划落到两套 API」，而是**它落到同一套**：谁也不许
// 按 target 把搬运换成裸 `DataCopy`。两侧各写一遍而不是合成一条，是为了让「哪天又有人
// 给某个 target 加回落」这件事在两处都变红。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#output_desc = #afir.tensor_desc<axis = [0], role = <output>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
#load_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>
#output_queue = #afir.buffer_desc<space = <ub>, buffer_id = 3, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(%input: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.buffer_desc = #gm_buffer}, %output: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#input_desc, #output_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 10,
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %load_tile = memref.alloc() {afir.buffer_desc = #load_buffer} : memref<16xf16>
    %compute_tile = memref.alloc() {afir.buffer_desc = #output_queue} : memref<16xf16>
    %loaded = afir.load %input outs(%load_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#input_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed = afir.add %loaded, %loaded outs(%compute_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#compute_desc, #compute_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %computed outs(%output) {
      afir.destination_descs = [#output_desc],
      afir.operand_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// REGBASE: // -----KERNEL_BODY-----
// The fixed include block comes first, extension headers after it, so an
// extension header can rely on kernel_operator.h having been included.
// REGBASE: #include "kernel_operator.h"
// REGBASE: #include "autofuse_tiling_data.h"
// REGBASE: #include <cstdint>
// REGBASE: #include "datacopy_reg_base.h"
// The wrapper is a free function, not an `AscendC::` member: an extension API
// keeps whatever namespace its header declares it in.
// REGBASE: {{^ *}}DataCopyPadExtend<half, AscendC::PaddingMode::Normal>
// REGBASE: AscendC::Add
// store 也走包装。`-NOT` 放在最后一条正向 check 之后，管的是从 `Add` 到文件尾这一段，
// 正好是 store 的位置；load 那一处由上面那条正向 check 管。
// REGBASE-NOT: AscendC::DataCopy(

// MEMBASED: // -----KERNEL_BODY-----
// MEMBASED: #include "kernel_operator.h"
// MEMBASED: #include "autofuse_tiling_data.h"
// MEMBASED: #include <cstdint>
// MEMBASED: #include "datacopy.h"
// MEMBASED: {{^ *}}DataCopyPadExtend<half>
// MEMBASED: AscendC::Add
// MEMBASED-NOT: AscendC::DataCopy(
