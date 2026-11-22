// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.reduction_domain` 的 negative 用例。
//
// 这四条的共同点：没有任何下游会发现它们。分母是 `Π sizes` 一个乘积，空集的积是 1
// （等于不除）、少一个 size 的积缺一个因子、重复的 id 把同一根轴数两遍——三者都算得"成功"，
// 只是 Mean 的结果错了，而错的倍数没有规律、也不越界。只写字段的唯一门禁就是产出时这一道。

// RUN: af-opt --split-input-file --verify-diagnostics %s

// 空域：`N` 会变成 1，也就是"不除"——Mean 静默退化成 Sum。
//
// 文本形态下它连解析都过不去（ODS 的逗号分隔列表至少要一项），所以这里报的是解析错误而不是
// verifier 那句 `must name at least one axis`。verifier 里那一条仍然留着：它管的是解析之外
// 的构造路（`getChecked`），而写这个属性的 `annotateReductionDomains` 走的是程序构造。
// expected-error@+2 {{expected integer value}}
// expected-error@+1 {{failed to parse ReductionDomainAttr parameter 'axes'}}
module attributes {afir.reduction_domain = #afir.reduction_domain<axes = [], sizes = []>} {
}

// -----

// 轴与尺寸不等长：乘积少一个因子，分母偏小。
// expected-error@+1 {{reduction_domain has 2 axes but 1 sizes}}
module attributes {afir.reduction_domain = #afir.reduction_domain<axes = [1, 3], sizes = [<"32">]>} {
}

// -----

// 同一根轴数两遍：这正是切分出来的 inner/outer 指向同一来源时会犯的错，分母变成 `R^2` 量级。
// expected-error@+1 {{reduction_domain repeats axis 1}}
module attributes {afir.reduction_domain = #afir.reduction_domain<axes = [1, 1], sizes = [<"32">, <"8">]>} {
}

// -----

// 负轴号：轴 id 是 `#afir.asc_graph` 里的下标，负数查不到任何一根轴。
// expected-error@+1 {{reduction_domain axis must be non-negative}}
module attributes {afir.reduction_domain = #afir.reduction_domain<axes = [-1], sizes = [<"32">]>} {
}
