/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_BASIC_VCONV_H
#define ASCIR_TARGET_ASC_BASIC_VCONV_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

// ===----------------------------------------------------------------------===//
// Type conversion operations
// ===----------------------------------------------------------------------===//

template <typename OpType>
LogicalResult printCastL01Template(CodeEmitter& emitter, OpType op)
{
    auto& os = emitter.ostream();
    auto dstType = cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    auto srcType = cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstType));
    os << ", ";
    FAIL_OR(emitter.emitType(op.getLoc(), srcType));
    os << ", " << op.getIsSetMask() << ">";
    return success();
}

template <typename OpType>
LogicalResult printCastDeqL01Template(CodeEmitter& emitter, OpType op)
{
    auto& os = emitter.ostream();
    auto dstType = cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    auto srcType = cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstType));
    os << ", ";
    FAIL_OR(emitter.emitType(op.getLoc(), srcType));
    os << ", " << op.getIsSetMask() << ", " << op.getIsVecDeq() << ", " << op.getHalfBlock() << ">";
    return success();
}

template <typename OpType>
LogicalResult printCastDeqL2Template(CodeEmitter& emitter, OpType op)
{
    auto& os = emitter.ostream();
    auto dstType = cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    auto srcType = cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstType));
    os << ", ";
    FAIL_OR(emitter.emitType(op.getLoc(), srcType));
    os << ", " << op.getIsVecDeq() << ", " << op.getHalfBlock() << ">";
    return success();
}

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CastL0Op op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CastL1Op op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CastL2Op op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CastDeqL0Op op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CastDeqL1Op op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CastDeqL2Op op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::SetDeqScaleOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::SetDeqScaleL4Op op);
} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_BASIC_VCONV_H
