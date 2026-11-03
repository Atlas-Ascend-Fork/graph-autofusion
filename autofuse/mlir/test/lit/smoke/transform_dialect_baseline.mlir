// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// transform dialect 基线验证。
//
// 设计文档 3.4 计划把 optimize/autoschedule 迁移为 transform.afir.*；3.0 的待确认项
// 表把「transform dialect 在当前 LLVM 基线可用且可扩展」列为实施前必须先验证的前提，
// 它决定 5.1 步 9 是否可行。本用例把该结论固定成本仓构建里的可执行事实，而不是靠
// 读上游文档推断——LLVM 基线若换成不带 transform dialect 的构建，这里立即失败。
//
// 验证三件事：
//   1. transform 方言可解析、可注册（af-opt 的 DialectRegistry）；
//   2. --transform-interpreter 能在 AFIR IR 上实际执行；
//   3. 匹配 + 改写这一对基本能力可用——foreach_match 按 Op 名匹配 afir.add，
//      再由 action sequence 改写命中的 Op。这正是调度变换所需的最小能力集。
//
// 注意用例刻意用 schedule.selected 而非 afir.selected 打标：AFIR Op 带
// AFIRRejectUnknownAttributes，任何未登记的 afir.* discardable 属性都会被
// verifier 拒绝。迁移自定义 transform.afir.* Op 时，凡是要往 AFIR Op 上写属性的
// 变换，属性名都必须先在 ODS 或已知表里登记。

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
