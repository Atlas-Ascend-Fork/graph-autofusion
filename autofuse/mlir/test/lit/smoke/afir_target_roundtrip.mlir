// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.target` 的 round-trip。
//
// 此前这份信息是 `ascendc.target_aiv_num` / `ascendc.target_ub_size` 两个散的 i64，
// 挂在 `api/ascendc` 注册的方言前缀下，因而既没有 verifier 也没有文本形态可锁；目标
// 标识（soc）则根本没进过 IR，一份 dump 出来的 plan module 看不出它是为哪颗芯片切的。
//
// 锁住文本有两层用处：bridge 的产出与 artifact 的消费之间有一个不依赖任何 pass 的
// 参照；`soc` 将来要被 `resolveSessionTargetProfile` 反解回 profile，靠的也是这个形态。

// RUN: af-opt %s | FileCheck %s
// RUN: af-opt %s | af-opt | FileCheck %s

// CHECK: module attributes {afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>}
module attributes {afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>} {
}
