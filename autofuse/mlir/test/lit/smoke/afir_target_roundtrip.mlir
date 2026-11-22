// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.target` 的 round-trip。
//
// target 将 soc 标识、核数和 UB 大小收敛为有类型属性，替代分散的后端整数属性。
// round-trip 检查其文本形式，使 dump 能保留完整的目标信息。
//
// 固定序列化形式，为 bridge 产出和 artifact 消费提供独立基线，
// 并保留后续按 soc 恢复目标 profile 所需的信息。

// RUN: af-opt %s | FileCheck %s
// RUN: af-opt %s | af-opt | FileCheck %s

// CHECK: module attributes {afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>}
module attributes {afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>} {
}
