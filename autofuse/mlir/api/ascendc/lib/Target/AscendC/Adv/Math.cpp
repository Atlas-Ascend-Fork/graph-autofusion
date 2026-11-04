/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Adv/Math.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Other math library operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ExpOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    auto resultType = op.getDst().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), resultType));
    os << ", " << emitter.getOrCreateName(op.getTaylorExpandLevel()) << ", "
       << emitter.getOrCreateName(op.getIsReuseSource()) << ">";
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc());
    if (auto sharedTmpBuffer = op.getSharedTmpBuffer()) {
        os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
    }
    os << ", " << emitter.getOrCreateName(op.getCalCount()) << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::AxpyOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    auto dstType = op.getDst().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), dstType));
    os << ", ";
    auto srcType = op.getSrc().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), srcType));
    os << ", " << emitter.getOrCreateName(op.getIsReuseSource()) << ">";
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getScalar());
    if (auto sharedTmpBuffer = op.getSharedTmpBuffer()) {
        os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
    }
    os << ", " << emitter.getOrCreateName(op.getCalCount()) << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CumSumOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getLastRow()) << ", " << emitter.getOrCreateName(op.getSrc());
    if (auto sharedTmpBuffer = op.getSharedTmpBuffer()) {
        os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
    }
    os << ", AscendC::CumSumInfo(" << emitter.getOrCreateName(op.getLastAxis()) << ", "
       << emitter.getOrCreateName(op.getReuseSource()) << ", " << emitter.getOrCreateName(op.getOutputLastRow())
       << "))";

    return success();
}
