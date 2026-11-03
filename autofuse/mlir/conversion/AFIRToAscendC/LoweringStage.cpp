/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/LoweringStage.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::ascend::kernel {

bool hasMaterializedAscendCResources(func::FuncOp func) {
  WalkResult found = func.walk([](Operation *op) {
    return isa<ascendc::AscendCDialect>(op->getDialect()) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return found.wasInterrupted();
}

bool hasPipelineSynchronization(func::FuncOp func) {
  WalkResult found = func.walk([](Operation *op) {
    // AFIR 同步族与 AscendC 的队列算子都算数：同步阶段的产物现在分两层——本层的顺序
    // 由 `afir.pipe_barrier` / `event_record` / `event_wait` 表达，队列路径仍落在
    // `ascendc` 的 enque / deque 上（退出 TQue 依赖 ADR-11/12 的 pool 形态）。判据要认
    // 全部，否则第二次跑会在已经同步过的函数上重插一遍。
    return isa<afir::PipeBarrierOp, afir::EventRecordOp, afir::EventWaitOp, afir::BarrierAllOp, afir::SyncAllOp,
               ascendc::TQueBindEnqueTensorOp, ascendc::TQueBindDequeTensorOp, ascendc::PipeBarrierOp>(op)
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return found.wasInterrupted();
}

}  // namespace mlir::ascend::kernel
