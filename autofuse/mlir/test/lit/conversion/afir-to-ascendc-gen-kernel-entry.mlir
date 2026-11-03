// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-gen-kernel-entry %s | af-opt | FileCheck %s

// This fixture covers both stage-internal template dispatch (scf.index_switch
// on stage0 with 2 templates) and inter-stage chaining (sync_all + call to
// stage1_template0).
module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [2, 1]>,
  ascendc.tiling_schema = !emitasc.py_struct<"plan0TilingData", [
       !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>,
       !emitasc.py_struct<"AscGraph0ScheduleResult0G1TilingData", [ui32], ["tiling_key"]>],
      ["graph0_result0_g0_tiling_data", "graph0_result0_g1_tiling_data"]>
} {
  func.func private @stage0_template0(
       %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 10, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
         ascendc.num_inputs = 0 : i32,
         afplan.variant = #afplan.variant<stage = 0, template = 0>
      } {
    return
  }
  func.func private @stage0_template1(
       %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 20, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
         ascendc.num_inputs = 0 : i32,
         afplan.variant = #afplan.variant<stage = 0, template = 1>
      } {
    return
  }
  func.func private @stage1_template0(
       %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult0G1TilingData", [ui32], ["tiling_key"]>)
      attributes {
        afir.asc_graph = #afir.asc_graph<tiling_key = 30, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
         ascendc.num_inputs = 0 : i32,
         ascendc.need_sync_all, afplan.variant = #afplan.variant<stage = 1, template = 0>
      } {
    return
  }
}

// `_main` 一 stage 收一个 tiling 形参，序号即 stage：ATT 的最外层 TilingData 里没有 plan
// 那一层，各 stage 的结构体平铺在根上，所以这里没有「先从 plan tiling 上取成员」这一步，
// 每个 stage 直接用自己那个形参。
// CHECK-LABEL: module @plan0
// CHECK: func.func @plan0_main
// CHECK-SAME: %[[G0:[A-Za-z0-9_]+]]: !emitasc.py_struct<"AscGraph0ScheduleResult0G0TilingData"
// CHECK-SAME: #afir.kernel_arg<role = tiling, index = 0>
// CHECK-SAME: %[[G1:[A-Za-z0-9_]+]]: !emitasc.py_struct<"AscGraph0ScheduleResult0G1TilingData"
// CHECK-SAME: #afir.kernel_arg<role = tiling, index = 1>
// CHECK-NOT: emitasc.member {{.*}} "graph0_result0_tiling_data"
// CHECK: emitasc.member %[[G0]] "tiling_key"
// CHECK: scf.index_switch
// CHECK: case 10
// CHECK: func.call @stage0_template0(%[[G0]])
// CHECK: case 20
// CHECK: func.call @stage0_template1(%[[G0]])
// 缺省分支必须陷入而不是空转：`tiling_key` 由 host 侧 tiling 写，落不进已下降的模板编号就
// 说明两侧对模板集的认知不一致。空 `scf.yield` 会静默跳过这个 stage，后面的 stage 照常跑在
// 没算过的数据上——数值错且没有任何报错。每个 stage 的分派表各查一次。
// CHECK: default {
// CHECK-NEXT: ascendc.trap
// CHECK: emitasc.member %[[G1]] "tiling_key"
// CHECK: func.call @stage1_template0(%[[G1]])
// CHECK: ascendc.sync_all_hard
// CHECK: default {
// CHECK-NEXT: ascendc.trap
