/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/Arith.h"

using namespace mlir;

namespace {
constexpr uint32_t bitWidth32 = 32;
constexpr uint32_t bitWidth16 = 16;
} // namespace

uint32_t mlir::getResultBitWidth(Type type)
{
    return isa<IndexType>(type) ? bitWidth32 : type.getIntOrFloatBitWidth();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, arith::AddUIExtendedOp op)
{
    auto& os = emitter.ostream();
    auto lhs = emitter.getOrCreateName(op.getLhs());
    auto rhs = emitter.getOrCreateName(op.getRhs());
    auto sum = op->getResult(0);
    if (failed(emitter.emitVariableDeclaration(sum, false))) {
        return failure();
    }
    os << " = " << lhs << " + " << rhs << ";\n";
    auto overflow = op->getResult(1);
    if (failed(emitter.emitVariableDeclaration(overflow, false))) {
        return failure();
    }
    // For unsigned addition, overflow is equivalent to the wrapped sum being
    // smaller than either operand.
    os << " = (" << emitter.getOrCreateName(sum) << " < " << lhs << ")";
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, arith::MulSIExtendedOp op)
{
    auto& os = emitter.ostream();
    auto resultType = op->getResult(1).getType();
    auto typeSize = getResultBitWidth(resultType);
    auto lhs = emitter.getOrCreateName(op.getLhs());
    auto rhs = emitter.getOrCreateName(op.getRhs());
    if (failed(emitter.emitVariableDeclaration(op->getResult(0), false))) {
        return failure();
    }
    // Widen to the next signed width before multiplying: at the narrow
    // (result) width, lhs * rhs is signed multiplication that can overflow
    // (UB) for exactly the operands this op exists to handle, i.e. whenever
    // the true product doesn't fit back in that width. Truncating a
    // wide-computed value back down on assignment is a well-defined
    // narrowing conversion, unlike overflow during the multiply itself.
    if (typeSize == bitWidth32) {
        os << " = static_cast<int64_t>(" << lhs << ") * static_cast<int64_t>(" << rhs << ");\n";
    } else if (typeSize == bitWidth16) {
        os << " = static_cast<int32_t>(" << lhs << ") * static_cast<int32_t>(" << rhs << ");\n";
    } else {
        llvm_unreachable("not implemented");
    }
    if (failed(emitter.emitVariableDeclaration(op->getResult(1), false))) {
        return failure();
    }
    if (typeSize == bitWidth32) {
        os << " = (static_cast<int64_t>(" << lhs << ") * static_cast<int64_t>(" << rhs << ")) >> 32";
    } else if (typeSize == bitWidth16) {
        os << " = (static_cast<int32_t>(" << lhs << ") * static_cast<int32_t>(" << rhs << ")) >> 16";
    } else {
        llvm_unreachable("not implemented");
    }

    return success();
}
