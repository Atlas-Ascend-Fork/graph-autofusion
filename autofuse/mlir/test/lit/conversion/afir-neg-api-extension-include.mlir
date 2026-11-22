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

// afir-api-extension-include.mlir 的第二个用例，钉的是同一条通道最不显眼的那一段：
// 两张支持表都有、名字不同的扩展头。
//
// `afir.neg` 在两个 target 上生成的调用文本逐字相同（`NegExtend<half>(...)`），
// 差异只在内联进来的是 `neg.h` 还是 `neg_reg_base.h`。afir-neg-api-family.mlir 走的是
// `--afir-to-ascendc-pipeline | ascir-translate`，不经过 AddKernelIncludes，所以那份
// 文件对头文件名一个字都没验到——把两个 implementation 的家族写反，它照样全绿。

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
    %computed = afir.neg %loaded outs(%compute_tile) {
      afir.destination_descs = [#compute_desc],
      afir.operand_descs = [#compute_desc],
      afir.result_descs = [#compute_desc],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
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
// 固定块在前，扩展头在后；`using namespace AscendC;` 是扩展头的前导，legacy 无条件写
// 在每个 kernel 里，这里只在有扩展头时写。
// REGBASE: #include "kernel_operator.h"
// REGBASE: #include "autofuse_tiling_data.h"
// REGBASE: #include <cstdint>
// REGBASE: using namespace AscendC;
// REGBASE: #include "neg_reg_base.h"
// REGBASE-NOT: #include "neg.h"
// REGBASE: {{^ *}}NegExtend<half>

// MEMBASED: // -----KERNEL_BODY-----
// MEMBASED: #include "kernel_operator.h"
// MEMBASED: #include "autofuse_tiling_data.h"
// MEMBASED: #include <cstdint>
// MEMBASED: using namespace AscendC;
// MEMBASED: #include "neg.h"
// MEMBASED-NOT: #include "neg_reg_base.h"
// 调用文本两侧一模一样——所以光看它区分不出支持表，上面那两条 include 才是判据。
// MEMBASED: {{^ *}}NegExtend<half>
