/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_SYNCHRONIZATION_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_SYNCHRONIZATION_H

#include "mlir/IR/Value.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include <optional>
#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::instruction {
class ImplementationRegistry;
class SelectionPolicyRegistry;
}  // namespace mlir::ascend::instruction

namespace mlir::ascend::kernel {

/// 函数里那个唯一的 `ascendc.pipe`（资源物化阶段建的），取不到时返回空值。
///
/// 事件槽的持有者是 `TPipe`，所以凡是要生成 `SetFlag` / `WaitFlag` 的下降点都得先找到
/// 它。不要就地再建一个：两个 `TPipe` 在生成的 C++ 里是两份声明，而槽位是按 pipe
/// 对象分配的。当前的取用点有两处——本文件的事件下降，和 `Broadcast.cpp` 的标量读同步。
Value findFunctionPipe(Operation *anchor);

/// 收集第 inputIndex 个输入所在存储的写入流水线。
///
/// 不按写入者的控制流位置过滤。materializeScalarRead 连续生成 SetFlag 和 WaitFlag，
/// 等待该流水线在同步点之前的任务完成。因此，只需确定可能写入该存储的流水线，无须证明
/// 写入者与读取者处于同一 block、条件分支或循环迭代。同步不负责保证输入数据已初始化。
///
/// 写入者包含 AFIRTransferOut 时返回失败，沿用既有的不支持行为。
/// 函数内没有写入者时，尝试按存储容器推断；仍无法确定时返回空集合。
FailureOr<SmallVector<afir::Pipe>> analyzeScalarWriterPipes(Operation *anchor, unsigned inputIndex);

/// 下降阶段仅读取 afir.scalar_read_sync；属性缺失或无效时返回空集合，由调用方报告匹配失败。
/// 不重新分析混合 IR，避免遗漏已下降的写入者，进而少生成同步事件。
SmallVector<ascendc::HardEvent> recordedScalarReadEvents(Operation *anchor, unsigned inputIndex);

/// 能力查询优先读取属性；属性缺失时直接分析尚未下降的 AFIR 计算算子。
/// 后者用于单独执行 --afir-select-implementation 的 lit 用例和能力探针。
SmallVector<ascendc::HardEvent> scalarReadEvents(Operation *anchor, unsigned inputIndex);

/// 逐条排空 `events` 里的事件，再读 `source` 的第 0 个元素。`events` 为空即失败。
FailureOr<Value> materializeScalarRead(Operation *anchor, Value source, ArrayRef<ascendc::HardEvent> events,
                                       RewriterBase &rewriter);

LogicalResult registerSynchronizationImplementations(ascend::instruction::ImplementationRegistry &registry,
                                                     ascend::instruction::SelectionPolicyRegistry *policies = nullptr);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_SYNCHRONIZATION_H
