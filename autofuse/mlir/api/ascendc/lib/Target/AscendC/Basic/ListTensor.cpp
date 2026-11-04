/*
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/ListTensor.h"

using namespace mlir;
using namespace mlir::ascendc;

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TensorDescOp op)
{
    auto& os = emitter.ostream();
    if (op->getNumResults() == 1) {
        FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
        os << " = ";
    }
    os << ascNamespace << "::TensorDesc<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getElementType()));
    os << ">()";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TensorDescSetShapeAddrOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getTensorDesc()) << "." << op.getAPIName() << "("
       << "(uint64_t* )" << emitter.getOrCreateName(op.getShapePtr()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ListTensorDescV2Op op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = "
       << "AscendC::ListTensorDesc(" << emitter.getOrCreateName(op.getData()) << ", "
       << emitter.getOrCreateName(op.getLength()) << ", " << emitter.getOrCreateName(op.getShapeSize()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ListTensorDescGetDataPtrOp op)
{
    auto& os = emitter.ostream();
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    os << " = " << emitter.getOrCreateName(op.getTensor()) << "." << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getDtype()));
    os << ">(" << emitter.getOrCreateName(op.getIndex()) << ")";
    return success();
}
