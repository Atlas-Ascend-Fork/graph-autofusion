/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Adv/Normalization.h"

using namespace mlir;
using namespace mlir::ascendc;

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::RmsNormOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getDst().getType().getElementType()));
    os << ", " << op.getBasicBlock() << ">(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getGamma());
    if (auto buf = op.getSharedTmpBuffer()) {
        os << ", " << emitter.getOrCreateName(buf);
    }
    os << ", " << emitter.getOrCreateName(op.getEpsilon()) << ", " << emitter.getOrCreateName(op.getTiling()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::LayerNormOp op)
{
    auto& os = emitter.ostream();
    os << "{\n";
    os.indent();
    os << "static constexpr " << ascNamespace << "::LayerNormConfig lnConfig"
       << " = {.isNoBeta = false, .isNoGamma = false, .isOnlyOutput = false, .isOutputRstd = "
       << (op.getOutputRstd() ? "true" : "false") << "};\n";
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getGamma().getType().getElementType()));
    os << ", ";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getDst().getType().getElementType()));
    os << ", false, lnConfig>(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getDstMean()) << ", " << emitter.getOrCreateName(op.getDstVarRstd()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getGamma()) << ", "
       << emitter.getOrCreateName(op.getBeta()) << ", " << emitter.getOrCreateName(op.getEpsilon()) << ", "
       << emitter.getOrCreateName(op.getSharedTmpBuffer()) << ", " << emitter.getOrCreateName(op.getPara()) << ", "
       << emitter.getOrCreateName(op.getSeparateTiling()) << ");\n";
    os.unindent() << "}\n";
    return success();
}
