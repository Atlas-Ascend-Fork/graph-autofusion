/*
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_BASIC_TERNARY_SCALAR_INSTR_H
#define ASCIR_TARGET_ASC_BASIC_TERNARY_SCALAR_INSTR_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Vector-ternary-scalar operations
//===----------------------------------------------------------------------===//

template <typename VecScalarL0Op>
auto printOperation(CodeEmitter& emitter, VecScalarL0Op op) -> LogicalResultForT<VecScalarL0Op, ascendc::AxpyL0Op>
{
    auto& os = emitter.ostream();
    auto dstTensorType = cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    auto srcTensorType = cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstTensorType));
    os << ", ";
    FAIL_OR(emitter.emitType(op.getLoc(), srcTensorType));
    os << ", " << op.getIsSetMask() << ">";
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getScalar()) << ", " << emitter.getOrCreateName(op.getMask()) << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}

template <typename VecScalarL1Op>
auto printOperation(CodeEmitter& emitter, VecScalarL1Op op) -> LogicalResultForT<VecScalarL1Op, ascendc::AxpyL1Op>
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);
    auto dstTensorType = cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    auto srcTensorType = cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstTensorType));
    os << ", ";
    FAIL_OR(emitter.emitType(op.getLoc(), srcTensorType));
    os << ", " << op.getIsSetMask() << ">";
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getScalar()) << ", " << maskName << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}

template <typename VecScalarL2Op>
auto printOperation(CodeEmitter& emitter, VecScalarL2Op op) -> LogicalResultForT<VecScalarL2Op, ascendc::AxpyL2Op>
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName();
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getScalar()) << ", " << emitter.getOrCreateName(op.getCalCount()) << ")";
    return success();
}

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_BASIC_TERNARY_SCALAR_INSTR_H
