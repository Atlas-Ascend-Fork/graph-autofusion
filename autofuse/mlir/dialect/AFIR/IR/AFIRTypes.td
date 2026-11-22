// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- AFIRTypes.td - AFIR type definitions ----------------*- tablegen -*-===//
//
// AFIR type definitions.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TYPES
#define AFIR_TYPES

include "AFIRAttrTypeBase.td"
include "AFIREnums.td"

//===----------------------------------------------------------------------===//
// Token: one cross-pipe ordering edge, as an SSA value
//===----------------------------------------------------------------------===//

def AFIR_TokenType : AFIR_Type<"Token", "token"> {
  let summary = "A cross-pipe happens-before edge";
  let description = [{
    一条跨 pipe 的 happens-before 边，以 SSA 值的形式存在：由 `afir.event_record`
    产出、被 `afir.event_wait` 消费，两点之间的 use-def 链就是那条边本身。

    **为什么带 pipe 对**（ADR-19）：硬件与两个后端都按 pipe 对组织同步——AscendC 的
    `SetFlag` / `WaitFlag` 本就是一对 pipe。token 不带 pipe 信息，lowering 时就得重新
    把它推断回来，而 verifier 也无从发现配对错误：一条本该是「搬入 → 计算」的边被写成
    「计算 → 搬出」，在无类型 token 下是合法 IR，失败方式是静默的数据竞争。

    `from` 与 `to` 必须不同。同一条 pipe 内程序序即执行序，不需要事件；写成同 pipe 的
    token 说明作者对这条边的意图与它实际能保证的事情不符，因此直接拒绝，而不是当成
    一条无害的空边放过去。

    片上存储的分配与同步在 AFIR 里是解耦的（ADR-12：不建 queue 抽象）——存储归
    `memref`，顺序归本类型。AscendC 的 `TQue` 把两者捆在一起，代价是 TQueue 与 TBuffer
    的空间不能互相复用。
  }];

  let parameters = (ins "::mlir::afir::Pipe":$from, "::mlir::afir::Pipe":$to);
  let assemblyFormat = "`<` `from` `=` $from `,` `to` `=` $to `>`";
  let genVerifyDecl = 1;
}

#endif // AFIR_TYPES
