/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_UNIVERSALEMITTER_H
#define ASCIR_TARGET_ASC_UNIVERSALEMITTER_H

#include "mlir/Support/LogicalResult.h"

#include "ascir/Target/Asc/CodeEmitter.h"

namespace mlir {
namespace ascendc {
template <typename ConcreteOp>
LogicalResult emitFunctionParams(CodeEmitter& emitter, ConcreteOp op, size_t startPos = 0)
{
    auto& os = emitter.ostream();
    bool first = true;
    for (size_t i = startPos; i < op.getOperation()->getNumOperands(); ++i) {
        if (!first) {
            os << ", ";
        }
        first = false;
        auto operand = op.getOperation()->getOperand(i);
        os << emitter.getOrCreateName(operand);
    }
    return success();
}

template <typename ConcreteOp>
LogicalResult autoPrintConstructorOp(CodeEmitter& emitter, ConcreteOp op)
{
    return emitter.emitVariableDeclaration(op->getResult(0), false);
}

template <typename ConcreteOp>
LogicalResult autoPrintMemberFuncOp(CodeEmitter& emitter, ConcreteOp op)
{
    auto resNum = op.getOperation()->getNumResults();
    auto& os = emitter.ostream();
    if (resNum == 1) {
        FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
        os << " = ";
    }
    os << emitter.getOrCreateName(op.getOperation()->getOperand(0)) << ".";
    os << op.getAPIName() << "(";
    FAIL_OR(emitFunctionParams<ConcreteOp>(emitter, op, 1));
    os << ")";
    return success();
}

template <typename ConcreteOp>
LogicalResult autoPrintAscFuncOp(CodeEmitter& emitter, ConcreteOp op)
{
    auto resNum = op.getOperation()->getNumResults();
    auto& os = emitter.ostream();
    if (resNum == 1) {
        FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
        os << " = ";
    }
    os << ascNamespace << "::" << op.getAPIName() << "(";
    FAIL_OR(emitFunctionParams<ConcreteOp>(emitter, op));
    os << ")";
    return success();
}

template <typename ConcreteOp>
LogicalResult autoPrintOp(CodeEmitter& emitter, ConcreteOp op)
{
    if constexpr (ConcreteOp::template hasTrait<mlir::OpTrait::AscConstructorTrait>()) {
        return autoPrintConstructorOp<ConcreteOp>(emitter, op);
    } else if constexpr (ConcreteOp::template hasTrait<mlir::OpTrait::AscMemberFuncTrait>()) {
        return autoPrintMemberFuncOp<ConcreteOp>(emitter, op);
    } else if constexpr (ConcreteOp::template hasTrait<mlir::OpTrait::AscFuncTrait>()) {
        return autoPrintAscFuncOp<ConcreteOp>(emitter, op);
    }
    return failure();
}
} // namespace ascendc
} // namespace mlir
#endif