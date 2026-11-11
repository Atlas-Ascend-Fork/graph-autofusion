/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AFIRToAscendCLoweringStage.h"

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
    // 同步阶段同时产生 AFIR 屏障、事件和 AscendC 队列算子，需要全部识别以防重复插入。
    // 队列路径仍使用 AscendC enque / deque；替换 TQue 依赖 ADR-11/12 的 pool 形态。
    return isa<afir::PipeBarrierOp, afir::EventRecordOp, afir::EventWaitOp, afir::BarrierAllOp, afir::SyncAllOp,
               ascendc::TQueBindEnqueTensorOp, ascendc::TQueBindDequeTensorOp, ascendc::PipeBarrierOp>(op)
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return found.wasInterrupted();
}

}  // namespace mlir::ascend::kernel
