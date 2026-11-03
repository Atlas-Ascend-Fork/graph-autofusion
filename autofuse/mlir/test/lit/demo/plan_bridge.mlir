// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-afir-gen | FileCheck %s
// AFIR 不感知上层接入框架：整个模块里不得出现框架类型判断（红线 Graph-2，设计文档 I6 / 3.7）。
// RUN: af-afir-gen | FileCheck %s --check-prefix=NOFRAMEWORK
// RUN: af-afir-gen | FileCheck %s --check-prefix=NOARRAY

// One Plan is one module. Stage is represented only by
// stage/template attributes on flat private functions.
// CHECK: module @elewise_brc_plan0 attributes {
// CHECK-SAME: afir.origin_vars = [#afir.expr<"s0">]
// 候选身份归 afplan（设计文档 5.1 步 7）；afir.stage_count 与本行的长度是同一个事实，
// 已随该步删除。
// CHECK-SAME: afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>
// CHECK-SAME: afplan.stages = #afplan.stages<template_counts = [2, 1]>

// 值的规划结果挂在值的定义点，一个值一份（设计文档 5.1 步 8）：kernel 参数与函数结果
// 在各自的 arg / result attr 上，Op 的结果在 Op 自己身上。此前它是 4 个与操作数按下标
// 对齐的数组属性，同一个值在每个使用点各存一份。
// CHECK: func.func private @stage0_template0
// CHECK-SAME: afir.buffer_desc = #afir.buffer_desc<space = <gm>, buffer_id = 0, size_bytes_expr = <"(16 * s0)">, life_time_axis_id = -1>
// CHECK-SAME: afir.buffer_desc = #afir.buffer_desc<space = <gm>, buffer_id = 1, size_bytes_expr = <"16">, life_time_axis_id = -1>
// CHECK-SAME: afir.buffer_desc = #afir.buffer_desc<space = <gm>, buffer_id = 6, size_bytes_expr = <"(16 * s0)">, life_time_axis_id = -1>
// CHECK-SAME: afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],role = <input>>, #afir.tensor_desc<axis = [1],role = <input>>]
// CHECK-SAME: afir.asc_graph = #afir.asc_graph<
// CHECK-SAME: size = <"s0">
// CHECK-SAME: size_var = [<"s0">]
// CHECK-SAME: afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, expr = #afir.expr<"s0">, id = 0 : i64}]
// CHECK-SAME: afir.function_role = #afir.function_role<kernel_body>
// CHECK-SAME: afir.result_descs = [#afir.tensor_desc<axis = [0, 1],role = <output>>]
// CHECK-SAME: afplan.variant = #afplan.variant<stage = 0, template = 0>
// CHECK: arith.constant 0 : index
// CHECK: tensor.dim {afir.shape_expr = #afir.expr<"s0">}
// CHECK: afir.load
// CHECK-SAME: afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(16 * s0)">, life_time_axis_id = -1>
// CHECK-SAME: afir.operand_descs = [#afir.tensor_desc<axis = [0, 1],role = <input>>]
// CHECK-SAME: afir.result_descs = [#afir.tensor_desc<axis = [0, 1],role = <compute>>]
// CHECK-SAME: indexing_maps = [#{{.*}}, #{{.*}}]
// 旧的 4 个数组属性不再出现在任何地方。
// NOARRAY-NOT: afir.operand_buffer_descs
// NOARRAY-NOT: afir.result_buffer_descs
// NOARRAY-NOT: afir.destination_buffer_descs
// NOARRAY-NOT: afir.arg_buffer_descs
// CHECK: func.func private @stage0_template1
// CHECK-SAME: afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, expr = #afir.expr<"s0">, id = 0 : i64}]
// CHECK-SAME: afir.function_role = #afir.function_role<kernel_body>
// CHECK-SAME: afplan.variant = #afplan.variant<stage = 0, template = 1>
// CHECK: arith.constant 0 : index
// CHECK: tensor.dim {afir.shape_expr = #afir.expr<"s0">}
// CHECK: func.func private @stage1_template0
// CHECK-SAME: afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, expr = #afir.expr<"s0">, id = 0 : i64}]
// CHECK-SAME: afir.function_role = #afir.function_role<kernel_body>
// CHECK-SAME: afplan.variant = #afplan.variant<stage = 1, template = 0>
// CHECK: arith.constant 0 : index
// CHECK: tensor.dim {afir.shape_expr = #afir.expr<"s0">}

// NOFRAMEWORK-NOT: afir.is_inductor
