/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/External/ExternalMath.h"

using namespace mlir;

LogicalResult mlir::printOperation(CodeEmitter& emitter, math::FmaOp op)
{
    FAIL_OR(isScalarOperation(op));
    FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
    auto& os = emitter.ostream();
    auto lhs = emitter.getOrCreateName(op.getOperand(0));
    auto mhs = emitter.getOrCreateName(op.getOperand(1));
    auto rhs = emitter.getOrCreateName(op.getOperand(2));
    os << lhs << " * " << mhs << " + " << rhs;
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, math::CopySignOp op)
{
    FAIL_OR(isScalarOperation(op));
    FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
    auto lhs = emitter.getOrCreateName(op.getLhs());
    auto rhs = emitter.getOrCreateName(op.getRhs());
    auto expression = formatv("(({0} < 0 && {1} > 0) || ({0} > 0 && {1} < 0)) ? -{0} : {0}", lhs, rhs);
    emitter.ostream() << expression;
    return success();
}
