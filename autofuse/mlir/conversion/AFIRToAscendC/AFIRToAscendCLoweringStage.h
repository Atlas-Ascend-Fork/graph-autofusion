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
// 根据 IR 结构判断下降阶段的产物是否存在，不依赖 pass 执行标记。
//
// 原有 afir.memory_allocated、afir.ascendc_materialized、afir.pipeline_synchronized
// 和 AFIR 侧的 afir.loop_generated 记录 pass 状态，存在以下问题（设计文档 D2、D5）：
//   1. 状态与 IR 中的资源、同步算子重复，可能不一致；memory_allocated 和
//      ascendc_materialized 还曾在同一处写入，重复记录同一事实。
//   2. 执行标记曾列入 AFIR.h 的 isKnownAFIR 白名单，但既不属于 payload，也不属于
//      候选身份（分层要求见 2.2）。结构正确的手写 IR 不应因缺少未公开标记而被拒绝。
//   3. 诊断应描述缺少的 IR 结构，而不是只要求用户先运行某个内部 pass。
//
// 判据需要区分已有产物和待处理资源：重复执行会插入重复资源或同步，漏执行则让下游读到
// 未完成物化的 IR。
//
// 无资源时 AFIRMemoryAllocate 在 CanonicalResources.cpp 的 resources.empty() 分支
// 提前返回，不产生 ascendc 算子。因此 hasMaterializedAscendCResources 为假不代表
// 必须先运行资源物化；检查前置条件应使用 hasPendingResourceMaterialization。
// 原有 memory_allocated 由 commit() 无条件写入；改用结构判据后仍须允许无资源函数
// 通过 AFIRPipelineSync，不能将其误报为资源未物化。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_LOWERINGSTAGE_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_LOWERINGSTAGE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::ascend::kernel {

// 函数中是否存在资源物化产生的 AscendC 算子，如 pipe、tbuf、queue、global_tensor。
// 物化前的 canonical IR 只有 AFIR 和 memref/scf/arith；循环生成引入的是 emitasc
// 字段访问算子，不会使此判据提前成立。
bool hasMaterializedAscendCResources(func::FuncOp func);

// 函数中是否存在 AFIRPipelineSync 产生的 AFIR 同步算子或 AscendC 队列、屏障算子。
// tque_bind_alloc_tensor / free_tensor 由资源物化阶段产生，不能据此判断同步是否已插入。
bool hasPipelineSynchronization(func::FuncOp func);

// 待物化资源的判据只依赖 AFIR，声明在 dialect/AFIR/Transforms/AFIRLoweringStage.h。

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_LOWERINGSTAGE_H
