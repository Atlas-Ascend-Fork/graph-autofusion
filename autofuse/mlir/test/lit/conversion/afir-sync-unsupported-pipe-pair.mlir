// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 硬件上没有对应事件的 pipe 对。
//
// 七台引擎两两组合有 42 个有序对，AscendC 的 `HardEvent` 覆盖其中 33 个。剩下的不是
// 写错，也不是映射表写漏——是硬件确实没有这条事件。判据 C4 要求这种情况在**编译期**
// 给出原因，而不是生成一段查不到的枚举值。
//
// 判掉它的不是 `TokenType` 的 verifier：类型只管 `from != to`，哪些对有事件是**目标**
// 的事，同一份 IR 换个后端可以是合法的（ADR-20）。所以闸门在能力查询上——
// `PipePairCapabilityProvider` 查不到 `HardEvent` 就报 `UnsupportedTarget`，实现选择
// 阶段据此拒绝。

// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 | FileCheck %s

module {
  func.func @scalar_to_cube() {
    // CHECK: 'afir.event_record' op has no supported Ascend implementation
    %t = afir.event_record : !afir.token<from = scalar, to = cube>
    afir.event_wait %t : !afir.token<from = scalar, to = cube>
    return
  }
}
