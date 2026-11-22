/*
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/Scalar.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Scalar operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ScalarCastOp op)
{
    auto& os = emitter.ostream();
    FAIL_OR(emitter.emitType(op.getLoc(), op.getDtype()));
    os << " " << emitter.getOrCreateName(op.getValueOut()) << " = ";
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getValueIn().getType(), true));
    os << ", ";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getDtype()));
    os << ", ";
    os << ascNamespace << "::RoundMode::" << ascendc::stringifyEnum(op.getRoundMode()).upper();
    os << ">(";
    os << emitter.getOrCreateName(op.getValueIn());
    os << ")";
    return success();
}
