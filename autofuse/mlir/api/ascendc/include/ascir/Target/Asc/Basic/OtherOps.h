/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_BASIC_OTHER_OPS_H
#define ASCIR_TARGET_ASC_BASIC_OTHER_OPS_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Other operations
//===----------------------------------------------------------------------===//

template <typename CVOp>
LogicalResultForT<CVOp, ascendc::AscendIsAICOp, ascendc::AscendIsAIVOp> printOperation(CodeEmitter& emitter, CVOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = g_coreType == AscendC::";
    if (isa<ascendc::AscendIsAICOp>(op)) {
        os << "AIC";
    } else {
        os << "AIV";
    }
    return success();
}

LogicalResult printOperation(CodeEmitter& emitter, ascendc::ConstructOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::FftsCrossCoreSyncOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::GetMrgSortResultOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::MrgSortOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::SortOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::PopStackBufferOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::SetFftsBaseAddrOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::ResetMaskOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::FixpipeOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::FixpipeWithWorkspaceOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::GetStoreAtomicConfigOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::IfAICOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::IfAIVOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::YieldOp op);

template <typename FixpipeOp>
auto printFixpipeTemplate(CodeEmitter& emitter, FixpipeOp op)
{
    auto& os = emitter.ostream();
    auto dstType = getElementTypeOrSelf(op.getDst());
    auto srcType = cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstType));
    os << ", ";
    FAIL_OR(emitter.emitType(op.getLoc(), srcType));
    os << ", ";
    os << emitter.getOrCreateName(op.getFixpipeConfig());
    os << ">";
    return success();
}

} // namespace ascendc

LogicalResult printOperation(CodeEmitter& emitter, LLVM::UndefOp op);

} // namespace mlir

#endif // ASCIR_TARGET_ASC_BASIC_OTHER_OPS_H