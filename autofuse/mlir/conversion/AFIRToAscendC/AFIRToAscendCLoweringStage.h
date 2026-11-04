/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- LoweringStage.h - 下降阶段的结构判据 --------------------*- C++ -*-===//
//
// 「某个下降阶段是否已经发生」的判据。
//
// 这些问题此前由三个 IR 属性回答——`afir.memory_allocated`、
// `afir.ascendc_materialized`、`afir.pipeline_synchronized`，加上 AFIR 侧的
// `afir.loop_generated`，都是「记录哪个 pass 跑过」的标记（设计文档 D5：IR 不
// 自描述、pass 不可组合）。三条理由要求把它们换成对 IR 自身的判断：
//
//   1. **同一事实的第二份编码**（D2）。阶段的产物就摆在 IR 里：物化过的函数带
//      `ascendc` 方言的 op，同步过的函数带 enque / deque / pipe_barrier。属性只是
//      把这件事再写一遍，且两份可以不一致——`afir.memory_allocated` 与
//      `afir.ascendc_materialized` 甚至在同一行被一起写入，本身就是同一事实的两份。
//   2. **公开面上不该有内部状态**。四个标记都登记在 `AFIR.h` 的 `isKnownAFIR`
//      白名单里，也就是 `afir` 命名空间的一部分。按 2.2 的分层，pass 执行状态既不是
//      payload 也不是候选身份。一份形态完全正确、但没挂这些属性的手写 IR 会被判成
//      「没跑过」，而属性名不在任何对外文档里。
//   3. **诊断不可操作**。「requires afir-memory-allocate to run first」告诉用户的是
//      流水线内部的 pass 名；「函数里没有 ascendc 方言的算子」说的是 IR 本身。
//
// 判据必须宁可多做一遍、不可少做一遍：这些阶段的重复执行会插入重复的资源与同步，
// 而漏判「已经做过」的后果是重复插入，漏判「还没做」的后果是下游读到半成品。
//
// **边界：「必然产出」有一个退化形态。** 资源为空时 `AFIRMemoryAllocate` 提前返回，
// 一个 `ascendc` 算子都不产出（CanonicalResources.cpp 的 `resources.empty()` 分支），
// 于是 `hasMaterializedAscendCResources` 在一个**确实跑过**的函数上为假。因此它只能
// 回答「物化过没有」，不能直接拿来当「该跑而没跑」的前置——后者要问的是
// `hasPendingResourceMaterialization`。旧的 `afir.memory_allocated` 由 `commit()`
// 无条件写入，没有这个退化形态；换成结构判据时这条边界必须显式处理，否则一个没有片上
// 资源的函数会在 `AFIRPipelineSync` 处硬报错，而它原先是成功的。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_LOWERINGSTAGE_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_LOWERINGSTAGE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::ascend::kernel {

// 片上资源是否已经物化成 AscendC 形态。
//
// 判据是「函数里出现了 `ascendc` 方言的算子」——这正是 `AFIRMemoryAllocate` 的产物
// （`ascendc.pipe` / `ascendc.tbuf` / `ascendc.queue` / `ascendc.global_tensor` 等）。
// 在它之前，canonical 形态只有 AFIR 与 memref/scf/arith；循环生成引入的是 `emitasc`
// 的取字段算子，不是 `ascendc`，所以这个判据不会提前变真。
bool hasMaterializedAscendCResources(func::FuncOp func);

// 流水同步是否已经插入。
//
// 判据是队列的入队 / 出队与 pipe barrier——`AFIRPipelineSync` 是它们唯一的产出点。
// 注意不能用 `ascendc.tque_bind_alloc_tensor` / `free_tensor`：那两个由资源物化阶段
// 产出，同步之前就已经在了。
bool hasPipelineSynchronization(func::FuncOp func);

// 「资源物化还欠债吗」是纯 AFIR 判据，住在 dialect/AFIR/Transforms/AFIRLoweringStage.h。

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_LOWERINGSTAGE_H
