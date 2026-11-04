/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_MLIR_MATH_H
#define ASCIR_TARGET_ASC_MLIR_MATH_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {

template <typename UnaryOpType>
auto printOperation(CodeEmitter& emitter, UnaryOpType op) -> LogicalResultForT<
    UnaryOpType, math::AbsFOp, math::SqrtOp, math::ExpOp, math::LogOp, math::CosOp, math::SinOp, math::Log2Op,
    math::ErfOp, math::CeilOp, math::FloorOp, math::RsqrtOp, math::Exp2Op, math::RoundOp>
{
    if (failed(isScalarOperation(op)) || failed(emitter.emitAssignPrefix(*op.getOperation()))) {
        return failure();
    }
    auto& os = emitter.ostream();
    if constexpr (std::is_same_v<UnaryOpType, math::AbsFOp>) {
        auto lhs = emitter.getOrCreateName(op.getOperand());
        os << "(" << lhs << " > static_cast<";
        FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
        os << ">(0)) ? " << lhs << " : -" << lhs;
    } else if constexpr (std::is_same_v<UnaryOpType, math::SqrtOp>) {
        os << "sqrt(" << emitter.getOrCreateName(op.getOperand()) << ")";
    } else if constexpr (std::is_same_v<UnaryOpType, math::ExpOp>) {
        os << ascNamespace << "::Exp(" << emitter.getOrCreateName(op.getOperand()) << ")";
    } else if constexpr (std::is_same_v<UnaryOpType, math::CosOp>) {
        os << ascNamespace << "::Cos(" << emitter.getOrCreateName(op.getOperand()) << ")";
    } else if constexpr (std::is_same_v<UnaryOpType, math::SinOp>) {
        os << ascNamespace << "::Sin(" << emitter.getOrCreateName(op.getOperand()) << ")";
    } else if constexpr (std::is_same_v<UnaryOpType, math::LogOp>) {
        os << ascNamespace << "::Log(" << emitter.getOrCreateName(op.getOperand()) << ")";
    } else if constexpr (std::is_same_v<UnaryOpType, math::Log2Op>) {
        os << ascNamespace << "::Log2(" << emitter.getOrCreateName(op.getOperand()) << ")";
    } else if constexpr (std::is_same_v<UnaryOpType, math::ErfOp>) {
        os << ascNamespace << "::Erf(" << emitter.getOrCreateName(op.getOperand()) << ")";
    } else if constexpr (std::is_same_v<UnaryOpType, math::CeilOp>) {
        os << ascNamespace << "::Ceil(" << emitter.getOrCreateName(op.getOperand()) << ")";
    } else if constexpr (std::is_same_v<UnaryOpType, math::FloorOp>) {
        os << ascNamespace << "::Floor(" << emitter.getOrCreateName(op.getOperand()) << ")";
    } else if constexpr (std::is_same_v<UnaryOpType, math::RsqrtOp>) {
        os << ascNamespace << "::Rsqrt(" << emitter.getOrCreateName(op.getOperand()) << ")";
    } else if constexpr (std::is_same_v<UnaryOpType, math::Exp2Op>) {
        os << ascNamespace << "::Exp(" << emitter.getOrCreateName(op.getOperand()) << " * Log(2))";
    } else if constexpr (std::is_same_v<UnaryOpType, math::RoundOp>) {
        os << ascNamespace << "::Round(" << emitter.getOrCreateName(op.getOperand()) << ")";
    } else {
        llvm_unreachable("not implemented");
    }
    return success();
}

template <typename BinaryOpType>
LogicalResultForT<BinaryOpType, math::Atan2Op> printOperation(CodeEmitter& emitter, BinaryOpType op)
{
    FAIL_OR(isScalarOperation(op));
    FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
    auto& os = emitter.ostream();
    if constexpr (std::is_same_v<BinaryOpType, math::Atan2Op>) {
        auto lhs = emitter.getOrCreateName(op.getLhs());
        auto rhs = emitter.getOrCreateName(op.getRhs());
        os << ascNamespace << "::Atan2(" << lhs << ", " << rhs << ")";
        return success();
    } else {
        llvm_unreachable("not implemented");
    }
    return failure();
}

LogicalResult printOperation(CodeEmitter& emitter, math::FmaOp op);

LogicalResult printOperation(CodeEmitter& emitter, math::CopySignOp op);

} // namespace mlir

#endif // ASCIR_TARGET_ASC_MLIR_MATH_H
