/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_UTILS_UTILS_H
#define ASCIR_DIALECT_UTILS_UTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include <cstddef>
#include <unordered_map>
#include <unordered_set>

namespace mlir {

template <typename T>
struct PointerLikeTypeHash {
    std::hash<const void*> h;
    size_t operator()(const T& op) const { return h(op.getAsOpaquePointer()); }
};

template <typename T>
using ValueMap = std::unordered_map<Value, T, PointerLikeTypeHash<Value>>;

using ValueSet = std::unordered_set<Value, PointerLikeTypeHash<Value>>;

using ValueVector = SmallVector<Value>;

namespace ascir {

inline LogicalResult eraseUnusedOp(Operation* op, PatternRewriter& rewriter)
{
    if (op->getUses().empty()) {
        rewriter.eraseOp(op);
        return success();
    }
    return failure();
}

} // namespace ascir
} // namespace mlir

#endif // ASCIR_DIALECT_UTILS_UTILS_H
