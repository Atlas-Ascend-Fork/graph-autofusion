// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// transform dialect 基线验证。
//
// 验证当前 LLVM 构建支持 transform dialect，为设计文档 3.4 的 transform.afir.*
// 调度迁移提供可执行基线。若依赖构建移除该方言，本用例应失败。
//
// 验证方言注册与解析、interpreter 在 AFIR 上执行，以及 foreach_match 按名称匹配
// 并由 action sequence 改写 afir.add。这些是调度变换需要的基础能力。
//
// 用 schedule.selected 标记，避免写入未登记的 afir.selected。
// AFIRRejectUnknownAttributes 会拒绝未知 afir.* 属性；自定义 transform.afir.*
// 若需写入该前缀属性，必须先在 ODS 或已知属性表登记。

// RUN: af-opt --transform-interpreter %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  func.func @payload(%lhs: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, size_bytes_expr = <"8">, life_time_axis_id = -1>}, %rhs: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, size_bytes_expr = <"8">, life_time_axis_id = -1>}, %out: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, size_bytes_expr = <"8">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>, #afir.tensor_desc<axis = [0], role = <input>>, #afir.tensor_desc<axis = [0], role = <output>>]
  } {
    %sum = afir.add %lhs, %rhs outs(%out) {
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>, #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>]
    } : (memref<4xf16>, memref<4xf16>), memref<4xf16> -> memref<4xf16>
    return
  }

  transform.named_sequence @match_afir_add(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %op ["afir.add"] : !transform.any_op
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @mark_selected(%op: !transform.any_op {transform.readonly}) {
    transform.annotate %op "schedule.selected" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.foreach_match in %root @match_afir_add -> @mark_selected
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @payload
// CHECK: afir.add
// CHECK-SAME: schedule.selected
