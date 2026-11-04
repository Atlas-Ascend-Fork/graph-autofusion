/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/BasicVecVconv.h"
#include "ascir/Target/Asc/Basic/BasicVecUnary.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Type conversion operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastL0Op op)
{
    auto& os = emitter.ostream();
    FAIL_OR(printCastL01Template(emitter, op));
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc())
       << ", AscendC::RoundMode::" << stringifyRoundMode(op.getRoundMode()) << ", "
       << emitter.getOrCreateName(op.getMask()) << ", " << emitter.getOrCreateName(op.getRepeatTimes()) << ", "
       << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);
    FAIL_OR(printCastL01Template(emitter, op));
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc())
       << ", AscendC::RoundMode::" << stringifyRoundMode(op.getRoundMode()) << ", " << maskName << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastL2Op op)
{
    auto& os = emitter.ostream();
    auto dstType = cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    auto srcType = cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstType));
    os << ", ";
    FAIL_OR(emitter.emitType(op.getLoc(), srcType));
    os << ">";
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << "AscendC::RoundMode::" << stringifyRoundMode(op.getRoundMode()) << ", "
       << emitter.getOrCreateName(op.getCalCount()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastDeqL0Op op)
{
    FAIL_OR(printCastDeqL01Template(emitter, op));
    printUnaryL0Params(emitter, op);
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastDeqL1Op op)
{
    auto maskName = printMask(emitter, op);
    FAIL_OR(printCastDeqL01Template(emitter, op));
    printUnaryL1Params(emitter, op, maskName);
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastDeqL2Op op)
{
    FAIL_OR(printCastDeqL2Template(emitter, op));
    printUnaryL2Params(emitter, op);
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SetDeqScaleL4Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getVdeq()) << ", "
       << emitter.getOrCreateName(op.getVdeqInfo()) << ")";

    return success();
}
