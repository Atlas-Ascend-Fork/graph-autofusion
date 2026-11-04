/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Fwk/TBuf.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Buffer operations (TBuf)
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TBufGetTensorOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = " << emitter.getOrCreateName(op.getBuffer()) << "." << op.getAPIName() << "<";
    Type elType = op.getTensor().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), elType));
    os << ">(";
    if (auto len = op.getLen()) {
        os << emitter.getOrCreateName(len);
    }
    os << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TBufGetWithOffsetOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = " << emitter.getOrCreateName(op.getBuffer()) << "." << op.getAPIName() << "<";
    Type elType = op.getTensor().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), elType));
    os << ">(" << emitter.getOrCreateName(op.getSize()) << ", " << emitter.getOrCreateName(op.getBufOffset()) << ")";
    return success();
}
