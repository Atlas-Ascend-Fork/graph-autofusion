/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Fwk/TQue.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// TQueQind operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TQueBindAllocTensorOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = " << emitter.getOrCreateName(op.getQueue()) << "." << op.getAPIName() << "<";
    Type elType = op.getTensor().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), elType));
    os << ">()";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TQueBindAllocTensorInPlaceOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getQueue()) << "." << op.getAPIName() << "<";
    Type elType = op.getTensor().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), elType));
    os << ">(" << emitter.getOrCreateName(op.getTensor()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TQueBindDequeTensorOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = " << emitter.getOrCreateName(op.getQueue()) << "." << op.getAPIName() << "<";
    Type elType = op.getTensor().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), elType));
    os << ">()";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TQueBindDequeTensorInPlaceOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getQueue()) << "." << op.getAPIName() << "<";
    Type elType = op.getTensor().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), elType));
    os << ">(" << emitter.getOrCreateName(op.getTensor()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TQueBindDequeTensorPosOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = " << emitter.getOrCreateName(op.getQueue()) << "." << op.getAPIName() << "<";
    CodeEmitter::emitTPosition(os, op.getSrcUserPos());
    os << ", ";
    CodeEmitter::emitTPosition(os, op.getDstUserPos());
    os << ", ";
    Type elType = op.getTensor().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), elType));
    os << ">()";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TQueBindEnqueTensorPosOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getQueue()) << "." << op.getAPIName() << "<";
    CodeEmitter::emitTPosition(os, op.getSrcUserPos());
    os << ", ";
    CodeEmitter::emitTPosition(os, op.getDstUserPos());
    os << ">(" << emitter.getOrCreateName(op.getTensor()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ToQueBindOp op)
{
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    auto& os = emitter.ostream();
    os << "& " << emitter.getOrCreateName(op.getResult()) << " = " << emitter.getOrCreateName(op.getOperand());
    return success();
}
