/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_INDEX_H
#define ASCIR_TARGET_ASC_AUTOFUSE_INDEX_H

#include "ascir/Target/Asc/Common.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

namespace mlir {

template <typename BinaryOpType>
auto printOperation(CodeEmitter& emitter, BinaryOpType op) -> LogicalResultForT<
    BinaryOpType, index::AddOp, index::MulOp, index::MaxUOp, index::MinUOp, index::RemUOp, index::CeilDivUOp,
    index::DivUOp>
{
    if (failed(isScalarOperation(op)) || failed(emitter.emitAssignPrefix(*op.getOperation()))) {
        return failure();
    }
    auto& os = emitter.ostream();
    auto lhs = emitter.getOrCreateName(op.getLhs());
    auto rhs = emitter.getOrCreateName(op.getRhs());
    if constexpr (std::is_same_v<BinaryOpType, index::CeilDivUOp>) {
        os << "(" << lhs << " + " << rhs << " - 1) / " << rhs;
        return success();
    }
    if constexpr (llvm::is_one_of<BinaryOpType, index::MaxUOp, index::MinUOp>::value) {
        os << "((" << lhs;
        os << (std::is_same_v<BinaryOpType, index::MaxUOp> ? " > " : " < ");
        os << rhs << ") ? (" << lhs << ") : (" << rhs << "))";
        return success();
    }
    os << lhs;
    if constexpr (std::is_same_v<BinaryOpType, index::AddOp>) {
        os << " + ";
    } else if constexpr (std::is_same_v<BinaryOpType, index::MulOp>) {
        os << " * ";
    } else if constexpr (std::is_same_v<BinaryOpType, index::DivUOp>) {
        os << " / ";
    } else if constexpr (std::is_same_v<BinaryOpType, index::RemUOp>) {
        os << " % ";
    } else {
        llvm_unreachable("not implemented");
    }
    os << rhs;
    return success();
}

LogicalResult printOperation(CodeEmitter& emitter, index::CmpOp op);

} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_INDEX_H
