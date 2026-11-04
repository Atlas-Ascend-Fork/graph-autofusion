/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Adv/Matmul.h"

using namespace mlir;
using namespace mlir::ascendc;

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::MatmulInitOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getMatmul()) << "." << op.getAPIName() << "(&"
       << emitter.getOrCreateName(op.getCubeTiling());
    if (auto pipe = op.getPipe()) {
        os << "," << emitter.getOrCreateName(pipe);
    }
    os << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::MatmulGetMatmulApiTilingOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = " << op.getAPIName();
    FAIL_OR(emitter.emitAscMatmulSimplifiedTemplate(op.getLoc(), op.getMatmulType(), false));
    os << "(" << emitter.getOrCreateName(op.getMmCFG()) << ", " << emitter.getOrCreateName(op.getL1Size()) << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::MatmulEndOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getMatmul()) << "." << op.getAPIName() << "()";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::RegistMatmulObjOp op)
{
    auto& os = emitter.ostream();
    os << "using namespace " << ascNamespace << ";\n";
    os << op.getAPIName() << "(&" << emitter.getOrCreateName(op.getPipe()) << ", "
       << emitter.getOrCreateName(op.getWorkspace()) << ", " << emitter.getOrCreateName(op.getMatmul());
    if (auto tiling = op.getTiling()) {
        os << ", &" << emitter.getOrCreateName(tiling);
    } else {
        auto matmulType = dyn_cast<ascendc::MatmulType>(op.getMatmul().getType());
        if (!matmulType)
            return op.emitError("expected matmul operand to have MatmulType");
        auto config = matmulType.getMatmulConfig();
        if (config.getSingleCoreM().getInt() > 0)
            os << ", (TCubeTiling*)nullptr";
    }
    os << ")";
    return success();
}
