// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afplan` 属性的 negative 用例（设计文档 5.1 步 7、5.3）。
//
// 迁移前，「候选编号必须非负」这条校验散在 PlanMetadata::validatePlanMetadata、
// TilingProgramSpec、GenAggregate* 各处，每个读点自己判一遍；而「名字写对了没有」
// 根本没人判——discardable 属性拼错只会被无声忽略，失败形态是候选分派走错分支。
//
// 收敛后两者都由方言一次钉死：值域在 Attr 的 verifier，名字与类型在
// AFPlanDialect::verifyOperationAttribute。

// RUN: af-opt --split-input-file --verify-diagnostics %s

module attributes {afplan.candidate = #afplan.candidate<subgraph = -1, plan = 0>} {
  // expected-error@-1 {{afplan.candidate subgraph must be non-negative}}
}

// -----

module attributes {afplan.candidate = #afplan.candidate<subgraph = 0, plan = -2>} {
  // expected-error@-1 {{afplan.candidate plan must be non-negative}}
}

// -----

module {
  // expected-error@+1 {{afplan.variant stage must be non-negative}}
  func.func private @bad_stage() attributes {afplan.variant = #afplan.variant<stage = -1, template = 0>} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{afplan.variant template must be non-negative}}
  func.func private @bad_template() attributes {afplan.variant = #afplan.variant<stage = 0, template = -1>} {
    return
  }
}

// -----

// 空分派表没有任何 stage 可分派；entry 生成器按它铺开 (stage, template) 表。
// 文本形态下 ODS 生成的解析器就要求列表至少有一个元素，轮不到 verifier；verifier 里
// 的同名检查仍然守着程序化构造（`StagesAttr::get`）这条路径。
module attributes {afplan.stages = #afplan.stages<template_counts = []>} {
  // expected-error@-1 {{expected integer value}}
  // expected-error@-2 {{failed to parse StagesAttr parameter 'template_counts'}}
}

// -----

// stage 存在但一个 template 都没有，同样无从分派。
module attributes {afplan.stages = #afplan.stages<template_counts = [2, 0]>} {
  // expected-error@-1 {{afplan.stages template count must be positive}}
}

// -----

// 名字写错：迁移前这条会被无声忽略，现在在方言闸门上判错。
// expected-error@+1 {{unknown afplan attribute 'afplan.candidates'}}
module attributes {afplan.candidates = #afplan.candidate<subgraph = 0, plan = 0>} {
}

// -----

// 名字对但类型不对：还原成整数属性即退回迁移前的无类型编码。
// expected-error@+1 {{afplan attribute 'afplan.candidate' has an unexpected value type}}
module attributes {afplan.candidate = 0 : i64} {
}
