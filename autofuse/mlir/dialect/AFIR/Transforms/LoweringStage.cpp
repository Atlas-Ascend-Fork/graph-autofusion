/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/Transforms/LoweringStage.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::afir {

bool hasPendingResourceMaterialization(func::FuncOp func) {
  WalkResult found = func.walk([](Operation *op) {
    if (!isa<AFIRDestinationStyleInterface>(op)) {
      return WalkResult::advance();
    }
    FailureOr<OperandPartition> partition = getOperandPartition(op);
    if (failed(partition) || partition->destinationCount == 0) {
      return WalkResult::advance();
    }
    for (Value destination : partition->destinations(op)) {
      if (isa<MemRefType>(destination.getType())) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found.wasInterrupted();
}

}  // namespace mlir::afir
