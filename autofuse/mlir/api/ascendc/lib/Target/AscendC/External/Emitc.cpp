/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/External/Emitc.h"

using namespace mlir;

LogicalResult mlir::printOperation(CodeEmitter& emitter, emitc::ConstantOp constantOp)
{
    Operation* operation = constantOp.getOperation();
    Attribute value = constantOp.getValue();

    return printConstantOp(emitter, operation, value);
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, emitc::VariableOp variableOp)
{
    Operation* operation = variableOp.getOperation();
    Attribute value = variableOp.getValue();

    return printConstantOp(emitter, operation, value);
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, emitc::CastOp castOp)
{
    raw_ostream& os = emitter.ostream();
    Operation& op = *castOp.getOperation();

    if (failed(emitter.emitAssignPrefix(op))) {
        return failure();
    }
    os << "(";
    if (failed(emitter.emitType(op.getLoc(), op.getResult(0).getType()))) {
        return failure();
    }
    os << ") ";
    os << emitter.getOrCreateName(castOp.getOperand());

    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, emitc::VerbatimOp verbatimOp)
{
    emitter.ostream() << verbatimOp.getValue();
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, emitc::IncludeOp includeOp)
{
    raw_ostream& os = emitter.ostream();

    os << "#include ";
    if (includeOp.getIsStandardInclude()) {
        os << "<" << includeOp.getInclude() << ">";
    } else {
        os << "\"" << includeOp.getInclude() << "\"";
    }

    return success();
}
