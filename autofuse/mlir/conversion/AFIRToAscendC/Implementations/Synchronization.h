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
/// 事件槽的持有者是 `TPipe`，所以凡是要发 `SetFlag` / `WaitFlag` 的下降点都得先找到
/// 它。**不要就地再建一个**：两个 `TPipe` 在生成的 C++ 里是两份声明，而槽位是按 pipe
/// 对象分配的。今天的取用点有两处——本文件的事件下降，和 `Broadcast.cpp` 的标量读同步。
Value findFunctionPipe(Operation *anchor);
/// Resolve the writer before this use; shared queue position alone is insufficient.
std::optional<ascendc::HardEvent> scalarReadEvent(Operation *anchor, Value source);
FailureOr<Value> materializeScalarRead(Operation *anchor, Value source, std::optional<ascendc::HardEvent> event,
                                       RewriterBase &rewriter);

LogicalResult registerSynchronizationImplementations(ascend::instruction::ImplementationRegistry &registry,
                                                     ascend::instruction::SelectionPolicyRegistry *policies = nullptr);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_SYNCHRONIZATION_H
