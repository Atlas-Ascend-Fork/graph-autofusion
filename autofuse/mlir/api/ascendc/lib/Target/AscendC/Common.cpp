/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Common.h"

using namespace mlir;

namespace {
constexpr uint32_t typeWidth16 = 16;
} // namespace

LogicalResult mlir::printConstantOp(CodeEmitter& emitter, Operation* operation, Attribute value)
{
    OpResult result = operation->getResult(0);
    auto& os = emitter.ostream();
    auto fType = dyn_cast_or_null<FloatType>(operation->getResult(0).getType());
    if (!fType || fType.getWidth() != typeWidth16) {
        os << "constexpr ";
    }
    // Emit a variable declaration for an emitc.constant op without value.
    if (auto oAttr = dyn_cast<emitc::OpaqueAttr>(value)) {
        if (oAttr.getValue().empty()) {
            // The semicolon gets printed by the emitOperation function.
            return emitter.emitVariableDeclaration(
                result,
                /*trailingSemicolon=*/false);
        }
    }

    // Emit a variable declaration.
    if (failed(emitter.emitAssignPrefix(*operation))) {
        return failure();
    }
    return emitter.emitAttribute(operation->getLoc(), value);
}

//===----------------------------------------------------------------------===//
// Mask operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SetVectorMaskL0Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getDtype(), true));
    os << ", " << ascNamespace << "::MaskMode::" << ascendc::stringifyEnum(op.getMode()).upper();
    os << ">(" << emitter.getOrCreateName(op.getLen()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SetVectorMaskL1Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getDtype(), true));
    os << ", " << ascNamespace << "::MaskMode::" << ascendc::stringifyEnum(op.getMode()).upper();
    os << ">(" << emitter.getOrCreateName(op.getMaskHigh()) << ", " << emitter.getOrCreateName(op.getMaskLow()) << ")";
    return success();
}
