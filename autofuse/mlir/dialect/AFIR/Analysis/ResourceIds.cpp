/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/Analysis/ResourceIds.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "mlir/IR/Value.h"

namespace mlir::afir {
namespace {

// 判据见头文件。
LogicalResult collectCanonicalResourceId(Operation *reporter, Value value, std::set<int64_t> &queueIds,
                                         std::set<int64_t> &bufferIds) {
  BufferDescAttr buffer = getBufferDescAttr(value);
  if (!buffer) {
    return success();
  }
  if (buffer.getSpace().getValue() == Space::GM) {
    return success();
  }
  if (buffer.getSpace().getValue() != Space::UB) {
    return reporter->emitError("canonical kernel resources support only GM and UB storage");
  }
  if (buffer.getBufferId() < 0) {
    reporter->emitError("canonical buffer descriptor is missing buffer_id");
    return failure();
  }
  (isCanonicalQueueResource(buffer) ? queueIds : bufferIds).insert(buffer.getBufferId());
  return success();
}

}  // namespace

bool isCanonicalQueueResource(BufferDescAttr buffer) {
  return buffer.getSlotCount() != -1 || buffer.getQueueDepth() != -1;
}

LogicalResult collectCanonicalResourceIds(func::FuncOp func, std::set<int64_t> &queueIds,
                                          std::set<int64_t> &bufferIds) {
  LogicalResult result = success();
  for (BlockArgument argument : func.getArguments()) {
    if (failed(result = collectCanonicalResourceId(func.getOperation(), argument, queueIds, bufferIds))) {
      return result;
    }
  }
  func.walk([&](Operation *operation) {
    if (failed(result) || !operation->hasAttr(kBufferDescAttrName) || operation->getNumResults() != 1) {
      return;
    }
    Value value = operation->getResult(0);
    result = collectCanonicalResourceId(operation, value, queueIds, bufferIds);
  });
  return result;
}

}  // namespace mlir::afir
