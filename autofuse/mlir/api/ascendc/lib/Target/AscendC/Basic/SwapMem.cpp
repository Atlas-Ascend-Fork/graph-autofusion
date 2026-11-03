/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/SwapMem.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Memory swap and workspace operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GetSysWorkspacePtrOp op)
{
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType(), true));
    auto& os = emitter.ostream();
    os << " " << emitter.getOrCreateName(op.getResult());
    os << " = " << op.getAPIName() << "()";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SetSysWorkspaceOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(";
    os << "reinterpret_cast<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getWorkspace().getType(), true));
    os << ">(" << emitter.getOrCreateName(op.getWorkspace()) << "))";
    return success();
}
