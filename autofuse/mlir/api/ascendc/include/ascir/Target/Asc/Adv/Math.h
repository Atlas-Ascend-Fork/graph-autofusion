/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_ADV_MATH_LIBS_H
#define ASCIR_TARGET_ASC_ADV_MATH_LIBS_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Unary math library operations
//===----------------------------------------------------------------------===//

template <typename UnaryMathOp>
auto printOperation(CodeEmitter& emitter, UnaryMathOp op) -> LogicalResultForT<
    UnaryMathOp, ascendc::AcoshOp, ascendc::AcosOp, ascendc::AsinhOp, ascendc::AsinOp, ascendc::AtanhOp,
    ascendc::AtanOp, ascendc::CeilOp, ascendc::CoshOp, ascendc::CosOp, ascendc::DigammaOp, ascendc::ErfcOp,
    ascendc::ErfOp, ascendc::FloorOp, ascendc::FracOp, ascendc::LgammaOp, ascendc::LogOp, ascendc::Log2Op,
    ascendc::RoundOp, ascendc::SignOp, ascendc::SinhOp, ascendc::SinOp, ascendc::TanhOp, ascendc::TanOp,
    ascendc::TruncOp>
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName();
    os << "<";
    auto dstType = op.getDst().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), dstType));
    os << ", " << emitter.getOrCreateName(op.getIsReuseSource()) << ">";
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc());
    if (auto sharedTmpBuffer = op.getSharedTmpBuffer()) {
        os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
    }
    if (auto calCount = op.getCalCount()) {
        os << ", " << emitter.getOrCreateName(calCount);
    }
    os << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// Binary math library operations
//===----------------------------------------------------------------------===//

template <typename BinaryMathOp>
LogicalResultForT<BinaryMathOp, ascendc::PowerOp, ascendc::XorOp> printOperation(CodeEmitter& emitter, BinaryMathOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName();
    os << "<";
    auto dstType = op.getDst().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), dstType));
    os << ", " << emitter.getOrCreateName(op.getIsReuseSource()) << ">";
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc0()) << ", "
       << emitter.getOrCreateName(op.getSrc1());
    if (auto sharedTmpBuffer = op.getSharedTmpBuffer()) {
        os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
    }
    if (auto calCount = op.getCalCount()) {
        os << ", " << emitter.getOrCreateName(calCount);
    }
    os << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// Other math library operations
//===----------------------------------------------------------------------===//

template <typename Clamp>
LogicalResultForT<Clamp, ascendc::ClampMaxOp, ascendc::ClampMinOp> printOperation(CodeEmitter& emitter, Clamp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName();
    os << "<";
    auto dstType = op.getDst().getType().getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), dstType));
    os << ", " << emitter.getOrCreateName(op.getIsReuseSource()) << ">";
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc());
    if (auto sharedTmpBuffer = op.getSharedTmpBuffer()) {
        os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
    }
    os << ", " << emitter.getOrCreateName(op.getScalar()) << ", " << emitter.getOrCreateName(op.getCalCount()) << ")";

    return success();
}

LogicalResult printOperation(CodeEmitter& emitter, ascendc::ExpOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::AxpyOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CumSumOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_ADV_MATH_LIBS_H
