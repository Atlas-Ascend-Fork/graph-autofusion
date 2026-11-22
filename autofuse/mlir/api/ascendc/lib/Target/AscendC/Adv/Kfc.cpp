/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Adv/Kfc.h"

using namespace mlir;
using namespace mlir::ascendc;

// Resource Management

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::KfcInitOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getServer()) << "." << op.getAPIName() << "("
       << emitter.getOrCreateName(op.getWorkspace()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::KfcInitObjOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getServer()) << "." << op.getAPIName() << "(&"
       << emitter.getOrCreateName(op.getPipe()) << ", ";
    llvm::interleaveComma(op.getOps(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::KfcIsRunOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = " << emitter.getOrCreateName(op.getServer()) << "." << op.getAPIName() << "()";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::KfcRunOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getServer()) << "." << op.getAPIName() << "("
       << emitter.getOrCreateName(op.getMatmul()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::KfcQuitOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getServer()) << "." << op.getAPIName() << "()";
    return success();
}
