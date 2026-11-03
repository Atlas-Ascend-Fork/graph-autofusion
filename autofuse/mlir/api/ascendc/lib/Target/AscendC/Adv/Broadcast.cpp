/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Adv/Broadcast.h"
#include "ascir/Target/Asc/Common.h"

using namespace mlir;
using namespace mlir::ascendc;

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::BroadcastOp op)
{
    auto& os = emitter.ostream();
    assert(op.getSrcShape().size() == op.getDstShape().size());
    auto dims = op.getSrcShape().size();
    os << "{\n";
    os.indent() << "const uint32_t dstShape[" << dims << "] = {";
    llvm::interleaveComma(op.getDstShape(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";
    os << "const uint32_t srcShape[" << dims << "] = {";
    llvm::interleaveComma(op.getSrcShape(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";
    os << ascNamespace << "::BroadcastTiling bcTiling;\n";
    os << ascNamespace << "::GetBroadcastTilingInfo<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getDst().getType().getElementType()));
    os << ", " << dims << ">(" << dims << ", dstShape, srcShape, false, bcTiling);\n";
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", dstShape, srcShape, &bcTiling);\n";
    os.unindent() << "}";
    return success();
}
