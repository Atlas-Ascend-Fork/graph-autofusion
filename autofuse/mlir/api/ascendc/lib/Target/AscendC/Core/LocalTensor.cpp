/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Core/LocalTensor.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// LocalTensor operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::LocalTensorV2Op op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    auto resultType = op.getResult().getType().getElementType();
    os << " = "
       << "AscendC::LocalTensor"
       << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), resultType));
    os << ">(";
    CodeEmitter::emitTPosition(os, op.getPos());
    os << ", " << emitter.getOrCreateName(op.getAddr()) << ", " << emitter.getOrCreateName(op.getTileSize()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::LocalTensorV3Op op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << "{";
    emitter.emitTPosition(os, op.getPos());
    os << ", " << op.getAddr() << ", " << op.getTileSize() << "}";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::LocalTensorReinterpretCastOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = " << emitter.getOrCreateName(op.getIn()) << "." << op.getAPIName() << "<";
    Type elType = op.getOut().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), elType));
    os << ">()";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::LocalTensorSubIndexOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = " << emitter.getOrCreateName(op.getTensor()) << "[" << emitter.getOrCreateName(op.getIndex()) << "]";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::LocalTensorBracketOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = " << emitter.getOrCreateName(op.getTensor()) << "(" << emitter.getOrCreateName(op.getIndex()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::LocalTensorGetPhyAddrV2Op op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = reinterpret_cast<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << ">(" << emitter.getOrCreateName(op.getTensor()) << "." << op.getAPIName() << "())";
    return success();
}
