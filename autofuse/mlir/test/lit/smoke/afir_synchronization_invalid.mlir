// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 同步族的 negative 用例。
//
// 错误同步可能只表现为运行期数据竞争，无法依靠 lowering 成功发现。
// 本文件验证能在类型和 verifier 阶段提前拒绝的同步错误。

// RUN: af-opt --split-input-file --verify-diagnostics %s

// 同 pipe 已由程序序保证执行顺序，不应创建同 pipe 的事件 token。
module {
  func.func @same_pipe() {
    // expected-error @below {{token pipes must differ, got 'vector' twice}}
    %t = afir.event_record : !afir.token<from = vector, to = vector>
    return
  }
}

// -----

// 一个 token 等两次。它代表一个具体的事件槽，第二次等的是一个已经消耗掉的事件，在
// 硬件上不会阻塞。
module {
  func.func @awaited_twice() {
    %t = afir.event_record : !afir.token<from = transfer_in, to = vector>
    // expected-error @below {{a token can only be awaited once}}
    afir.event_wait %t : !afir.token<from = transfer_in, to = vector>
    afir.event_wait %t : !afir.token<from = transfer_in, to = vector>
    return
  }
}

// -----

// 悬空的 record（token 无人消费）：下降层会 AllocEventID 占槽后无人 ReleaseEventID
// ——槽泄漏的失败形态是后续同型 record 拿到越界事件号。与「一个 token 只能等一次」
// 对称的弱化版（跨迭代携带的 token 有使用者，不在此列）。
module {
  func.func @unawaited_record() {
    // expected-error @below {{a recorded token must be consumed by an event_wait, directly or carried across iterations; a dead record leaks its event slot}}
    %t = afir.event_record : !afir.token<from = transfer_in, to = vector>
    return
  }
}

// -----

// pipe 名字写错。枚举化换来的正是这条。
module {
  func.func @unknown_pipe() {
    // expected-error @below {{expected one of [transfer_in, transfer_out, vector, transfer_local, cube, fixpipe, scalar]}}
    // expected-error @below {{failed to parse AFIR_TokenType parameter 'from'}}
    %t = afir.event_record : !afir.token<from = mte2, to = vector>
    return
  }
}

// -----

// 全序原语的驻地约束：barrier_all / sync_all 是 kernel 体内的全序/跨核事实，出现在
// 其它函数角色里没有意义。
module {
  func.func @barrier_all_outside_kernel_body() {
    // expected-error @below {{only allowed inside a kernel body function}}
    afir.barrier_all
    return
  }
}

// -----

module {
  func.func @sync_all_outside_kernel_body() {
    // expected-error @below {{only allowed inside a kernel body function}}
    afir.sync_all
    return
  }
}
