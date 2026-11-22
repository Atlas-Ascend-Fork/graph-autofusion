// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-gen-tiling-data %s | FileCheck %s

// This fixture covers: multi-stage, multi-template size_var merge, split
// size_var fields, queue/buffer fields (tmp buffer ids share the buffer table), and
// workspace fields.
module @plan3 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 3>,
  afplan.stages = #afplan.stages<template_counts = [2, 1]>,
  afir.workspace_tensor_ids = array<i64: 0, 1>
} {
  // stage0: 2 templates, split + queue/buffer + size_var
  func.func private @stage0_template0(%input: tensor<1024xf16>) attributes {
      afir.asc_graph = #afir.asc_graph<tiling_key = 10, axes = [
        <id = 0, name = "z1", axis_type = Original, size = #afir.expr<"z1_size">, align = #afir.expr<"16">>,
        <id = 1, name = "z1T", axis_type = TileOuter, size = #afir.expr<"Ceiling((z1_size / (z1t_size)))">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 2>,
        <id = 2, name = "z1t", axis_type = TileInner, size = #afir.expr<"z1t_size">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 1>], type = Compute,
        size_var = [#afir.expr<"z1_size">, #afir.expr<"z1t_size">]>,
      afir.queue_ids = array<i64: 0>, afir.buffer_ids = array<i64: 0, 1>,
      afir.api_tiling_fields = [{field = "api", type = "Api", fields = ["a", "b"]}],
      afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    // tmp buffer 与普通缓冲共用 id 空间，bridge 将其并入 afir.buffer_ids（此处为 1）。
    return
  }
  func.func private @stage0_template1(%input: tensor<1024xf16>) attributes {
      afir.asc_graph = #afir.asc_graph<tiling_key = 11, axes = [
        <id = 0, name = "z1", axis_type = Original, size = #afir.expr<"z1_size">, align = #afir.expr<"16">>,
        <id = 1, name = "z1T", axis_type = TileOuter, size = #afir.expr<"Ceiling((z1_size / (z1t_size)))">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 2>,
        <id = 2, name = "z1t", axis_type = TileInner, size = #afir.expr<"z1t_size">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 1>], type = Compute,
        size_var = [#afir.expr<"z1_size">, #afir.expr<"z1t_size">]>,
      afir.queue_ids = array<i64: 0>, afir.buffer_ids = array<i64: 0, 1>,
      afir.api_tiling_fields = [{field = "api", type = "Api", fields = ["a", "b"]}],
      afplan.variant = #afplan.variant<stage = 0, template = 1>
  } {
    return
  }
  // stage1: 1 template, different size_var
  func.func private @stage1_template0(%input: tensor<1024xf16>) attributes {
      afir.asc_graph = #afir.asc_graph<tiling_key = 20, axes = [
        <id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute,
        size_var = [#afir.expr<"block_size">]>,
      afplan.variant = #afplan.variant<stage = 1, template = 0>
  } {
    return
  }
}

// CHECK-LABEL: module @plan3 attributes {
// CHECK-SAME: afir.module_kind = "plan"
// CHECK-NOT: entry_name
// stage 结构体的类型名与它在最外层 TilingData 里的成员名都由 ATT 定死
// （`ScheduleGroupIdent::GetGroupPrefix()` / `GetItemPrefix()`，见
// `common/tiling_schema/schedule_group_names.h`），坐标取自 `#afplan.candidate`
// ——这里是 subgraph 0 / plan 3。`plan3TilingData` 只是载体，不会被生成 C++ 结构体。
// CHECK-SAME: ascendc.tiling_schema = !emitasc.py_struct<"plan3TilingData",
// CHECK-SAME: !emitasc.py_struct<"AscGraph0ScheduleResult3G0TilingData", [ui32, ui32, ui32, ui32, ui64, ui64,
// CHECK-SAME: !emitasc.py_struct<"Api", [ui32, ui32], ["a", "b"]>
// CHECK-SAME: "block_dim", "corenum", "ub_size", "hbm_size"
// CHECK-SAME: "workspace0", "workspace1"
// CHECK-SAME: "tiling_key"
// CHECK-SAME: "z1_size", "z1t_size"
// CHECK-SAME: "q0_size", "b0_size", "b1_size"
// CHECK-SAME: !emitasc.py_struct<"AscGraph0ScheduleResult3G1TilingData"
// CHECK-SAME: "block_dim", "corenum", "ub_size", "hbm_size"
// CHECK-SAME: "workspace0", "workspace1"
// CHECK-SAME: "tiling_key"
// CHECK-SAME: "block_size"
// CHECK-SAME: ["graph0_result3_g0_tiling_data", "graph0_result3_g1_tiling_data"]>
