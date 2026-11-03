/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/External/MemRef.h"

using namespace mlir;

LogicalResult mlir::printOperation(CodeEmitter& emitter, memref::AllocaOp op)
{
    auto mrType = op.getMemref().getType();
    FAIL_OR(emitter.emitType(op.getLoc(), mrType.getElementType(), op->hasAttr(ascendc::attr::emitAsUnsigned)));
    auto& os = emitter.ostream();
    os << " " << emitter.getOrCreateName(op.getResult());
    for (int64_t dim : mrType.getShape()) {
        os << "[" << dim << "]";
    }
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, memref::LoadOp op)
{
    if (failed(emitter.emitAssignPrefix(*op))) {
        return failure();
    }
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getMemref());
    for (Value index : op.getIndices()) {
        os << "[" << emitter.getOrCreateName(index) << "]";
    }
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, memref::StoreOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getMemref());
    for (Value index : op.getIndices()) {
        os << "[" << emitter.getOrCreateName(index) << "]";
    }
    os << " = " << emitter.getOrCreateName(op.getValueToStore());
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, memref::CastOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = reinterpret_cast<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << ">(" << emitter.getOrCreateName(op.getSource()) << ")";
    return success();
}
