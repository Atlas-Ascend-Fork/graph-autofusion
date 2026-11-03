// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#ifndef AFIR_SYNCHRONIZATION_OPS
#define AFIR_SYNCHRONIZATION_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "AFIRAttrs.td"
include "AFIRTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Cross-pipe ordering: record / wait
//===----------------------------------------------------------------------===//
//
// AFIR 放弃 AscendC 的 `TQue` 自动同步（ADR-12：存储归 memref，同步归 token/event），
// 因此正确性改由 AFIR 自己保证。这两个 Op 就是 AFIR 层表达顺序的唯一手段：本层此前
// 完全没有同步的表示形式——顺序是 `AFIRPipelineSync` 在 lowering 之后用 `TQue` 的
// enque/deque 补上的，那已经是 AscendC 的 IR（设计文档 3.2 / 5.1 步 8b 注记）。
//
// 二者以一个 `!afir.token` 相连，边的两端由 token 的 `from` / `to` 说明。事件槽分配
// （DAG coloring、profile 槽位上限、超限诊断）属于 lowering，见 3.6「同步的算法要点」。

def AFIR_EventRecordOp : AFIR_Op<"event_record",
    [MemoryEffects<[MemWrite<AFIR_PipelineResource>]>]> {
  let afirStability = AFIR_Experimental;
  let summary = "Record a cross-pipe ordering edge, producing a token";
  let description = [{
    在 `from` pipe 的当前位置记下一个事件，产出代表这条边的 token。语义是：本 Op 之前
    在 `from` pipe 上的一切，happens-before 消费该 token 的 `afir.event_wait` 之后在
    `to` pipe 上的一切。

    它不是「屏障」：`from` pipe 自己不会因此停下，`to` pipe 上没有 wait 的部分也不受
    影响。这正是显式同步相对 `TQue` 自动同步的收益——只挡住真正有依赖的那一对。

    产出的 token 至少要被一次 `afir.event_wait` 消费（直接或经循环携带）：没人用的
    record 会在下降层占下事件槽（`AllocEventID`）而无人归还（`ReleaseEventID`），槽
    泄漏的失败形态是后续同型 record 拿到越界事件号，因此由 verifier 拒绝。

    ```mlir
    afir.load ... // 搬入
    %t = afir.event_record : !afir.token<from = transfer_in, to = compute>
    afir.event_wait %t : !afir.token<from = transfer_in, to = compute>
    afir.add ...  // 计算，此时搬入已完成
    ```

    降到 AscendC 是 `SetFlag`；降到参照实现的对应原语亦然（附录 B）。
  }];

  let results = (outs AFIR_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($token)";
}

def AFIR_EventWaitOp : AFIR_Op<"event_wait",
    [MemoryEffects<[MemRead<AFIR_PipelineResource>, MemWrite<AFIR_PipelineResource>]>]> {
  let afirStability = AFIR_Experimental;
  let summary = "Wait for a recorded cross-pipe ordering edge";
  let description = [{
    在 token 的 `to` pipe 上等待 `afir.event_record` 记下的那个事件。`to` pipe 在此停下，
    直到 record 处 `from` pipe 的进度到达。

    一个 token 只能被等待一次：token 代表的是一个具体的事件槽，等两次意味着第二次等的
    是一个已经消耗掉的事件，在硬件上不会阻塞——失败方式是静默的数据竞争，因此由
    verifier 拒绝，而不是留给运行时。

    降到 AscendC 是 `WaitFlag`。
  }];

  let arguments = (ins AFIR_TokenType:$token);
  let assemblyFormat = "$token attr-dict `:` type($token)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Within-pipe ordering
//===----------------------------------------------------------------------===//

def AFIR_PipeBarrierOp : AFIR_Op<"pipe_barrier",
    [MemoryEffects<[MemWrite<AFIR_PipelineResource>]>]> {
  let afirStability = AFIR_Experimental;
  let summary = "Order all prior work on one pipe before all later work on it";
  let description = [{
    单条 pipe 内的全序点：`pipe` 上此前的一切完成之后，该 pipe 上此后的一切才开始。
    不跨 pipe——跨 pipe 的顺序只能由 `afir.event_record` / `afir.event_wait` 表达。

    AFIR 的依赖模型把「同一条 pipe 内程序序即执行序」当作公理，本 Op 是让该公理在目标
    上成立的那个手段：硬件允许同 pipe 内的部分重叠时，lowering 在需要处放一个
    `PipeBarrier`。因此它是 lowering 的产物，而不是前端该手写的东西——手写不会错，但
    多半意味着本该用一条 record/wait 边。
  }];

  let arguments = (ins PipeAttr:$pipe);
  // 前导的空字面量抑制 ODS 自己插的那个空格——`PipeAttr` 的打印器已经带了一个。
  let assemblyFormat = "``$pipe attr-dict";
}

//===----------------------------------------------------------------------===//
// Whole-ordering fallbacks
//===----------------------------------------------------------------------===//

def AFIR_BarrierAllOp : AFIR_Op<"barrier_all",
    [MemoryEffects<[MemWrite<AFIR_PipelineResource>]>]> {
  let afirStability = AFIR_Experimental;
  let summary = "Order all prior work on every pipe before all later work";
  let description = [{
    全引擎屏障：所有 pipe 上此前的一切完成之后，此后的一切才开始。它是
    `afir.pipe_barrier` 的跨 pipe 放宽——后者按设计只辖单 pipe，跨 pipe 顺序归
    record/wait；但「全序」语义（如一个 stage 的收尾兜底）没有必要逐对建边。降到
    AscendC 是 `PipeBarrier<PIPE_ALL>`。

    与 `afir.pipe_barrier` 同理，它是兜底原语，`AFIRPipelineSync` 不自动插入——自动
    插全序屏障与最小同步原则冲突。仅允许出现在 `KernelBody` 函数中。
  }];

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def AFIR_SyncAllOp : AFIR_Op<"sync_all",
    [MemoryEffects<[MemWrite<AFIR_PipelineResource>]>]> {
  let afirStability = AFIR_Experimental;
  let summary = "Cross-core barrier (hard variant)";
  let description = [{
    跨核栅栏：当前核在此停下，直到同组全部核到达。降到 AscendC 是 `AscendC::SyncAll()`
    （缺省 `isAIVOnly = true`，AIV-only 协议）——MLIR codegen 只覆盖非 cube（纯 AIV）图，
    `<false>` 是 AIC+AIV 混模协议，纯 AIV 发射下死锁。

    跨核顺序属调度层职责（单核函数体不辖跨核），因此 `AFIRPipelineSync` 不自动插入，
    仅供手写 IR 与后续 pass 使用。本 Op 是 hard 变体（全核参与）；soft 变体带
    `gmWorkspace` / `ubWorkspace` / `usedCores`，为 superkernel feed-sync 语义，登记为
    后续扩展。仅允许出现在 `KernelBody` 函数中。
  }];

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // AFIR_SYNCHRONIZATION_OPS
