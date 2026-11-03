// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afplan` 候选身份属性的 round-trip（设计文档 5.1 步 7、2.2 / 2.3、ADR-5）。
//
// 迁移前，这三件事写成 6 个挂在 `afir` 命名空间下的无类型整数属性：
//   afir.subgraph_id + afir.plan_id              -> #afplan.candidate
//   afir.stage_id    + afir.template_id          -> #afplan.variant
//   afir.template_counts_per_stage               -> #afplan.stages
//   afir.stage_count                             -> 删除（恒等于上一行的长度，全仓无读点）
//   afir.score_func                              -> afplan.score_func（名字迁移，仍是字符串）
//
// 两件事一起锁住：
//   1) 三个属性能原样解析并打印回来——候选身份的序列化形态是稳定契约；
//   2) `afir` 命名空间里不再出现这些字段（NOAFIR 前缀）——公开面不该看到内部编号。

// RUN: af-opt %s | af-opt | FileCheck %s
// RUN: af-opt %s | FileCheck %s --check-prefix=NOAFIR

module @plan attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 2>,
  afplan.stages = #afplan.stages<template_counts = [2, 1]>,
  afplan.score_func = "att_score"
} {
  func.func private @stage0_t0() attributes {afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    return
  }
  func.func private @stage0_t1() attributes {afplan.variant = #afplan.variant<stage = 0, template = 1>} {
    return
  }
  func.func private @stage1_t0() attributes {afplan.variant = #afplan.variant<stage = 1, template = 0>} {
    return
  }
}

// `afir.module_kind` 留在 afir：单候选时它仍决定生成哪种 module（判别见设计文档 2.3）。
// CHECK: module @plan attributes {
// CHECK-SAME: afir.module_kind = "plan"
// CHECK-SAME: afplan.candidate = #afplan.candidate<subgraph = 0, plan = 2>
// CHECK-SAME: afplan.score_func = "att_score"
// CHECK-SAME: afplan.stages = #afplan.stages<template_counts = [2, 1]>

// CHECK: func.func private @stage0_t0
// CHECK-SAME: afplan.variant = #afplan.variant<stage = 0, template = 0>
// CHECK: func.func private @stage0_t1
// CHECK-SAME: afplan.variant = #afplan.variant<stage = 0, template = 1>
// CHECK: func.func private @stage1_t0
// CHECK-SAME: afplan.variant = #afplan.variant<stage = 1, template = 0>

// NOAFIR-NOT: afir.subgraph_id
// NOAFIR-NOT: afir.plan_id
// NOAFIR-NOT: afir.stage_id
// NOAFIR-NOT: afir.template_id
// NOAFIR-NOT: afir.template_counts_per_stage
// NOAFIR-NOT: afir.stage_count
// NOAFIR-NOT: afir.score_func
