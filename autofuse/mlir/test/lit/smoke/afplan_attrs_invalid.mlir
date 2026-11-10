// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afplan` 属性的 negative 用例（设计文档 5.1 步 7、5.3）。
//
// 统一在属性层校验候选编号非负，在方言层校验属性名称与类型。
// 不能把拼错名称的候选字段作为 discardable 属性忽略，否则分派可能使用错误身份。
//
// 值域由 Attr verifier 检查；名字和类型由 AFPlanDialect::verifyOperationAttribute 检查。

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

// 每个分派表至少有一个 stage。文本解析器先拒绝空列表，
// Attr verifier 的对应分支由 StagesAttr 的程序化构造测试覆盖。
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

// 名字写错：迁移前这条会被无声忽略，现在在方言门禁上判错。
// expected-error@+1 {{unknown afplan attribute 'afplan.candidates'}}
module attributes {afplan.candidates = #afplan.candidate<subgraph = 0, plan = 0>} {
}

// -----

// 名字对但类型不对：还原成整数属性即退回迁移前的无类型编码。
// expected-error@+1 {{afplan attribute 'afplan.candidate' has an unexpected value type}}
module attributes {afplan.candidate = 0 : i64} {
}
