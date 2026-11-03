// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.reduction_domain` 的 round-trip。
//
// 这个属性是 Mean 分母的**唯一**合法来源：`axes` 是归约域对应的 Original 轴（归属），
// `sizes` 是与之一一对应的尺寸表达式，`N` 为其乘积。bridge 在来源还看得见的时候写，
// 下降侧只读与校验绑定（`afir::getProvenReductionDomain`）。
//
// 锁住文本形态的理由与 `#afir.target` 同类：它是 bridge 的产出与下降的消费之间一个不依赖
// 任何 pass 的参照。多一维、少一个 size、符号尺寸写成字面量，三者都会静默改变分母。

// RUN: af-opt %s | FileCheck %s
// RUN: af-opt %s | af-opt | FileCheck %s

// 一根轴、字面量尺寸：最常见的形态（`f32[16, 32]` 归约末维，N = 32）。
// CHECK: afir.reduction_domain = #afir.reduction_domain<axes = [1], sizes = [<"32">]>
module attributes {afir.reduction_domain = #afir.reduction_domain<axes = [1], sizes = [<"32">]>} {
}

// -----

// 多根独立归约轴 + 符号尺寸：`N` 是乘积，顺序进产物（物化出来的操作数序），所以顺序也必须
// round-trip。
// CHECK: afir.reduction_domain = #afir.reduction_domain<axes = [1, 3], sizes = [<"R0_org_size">, <"8">]>
module attributes {afir.reduction_domain = #afir.reduction_domain<axes = [1, 3], sizes = [<"R0_org_size">, <"8">]>} {
}
