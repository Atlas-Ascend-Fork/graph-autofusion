/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/VecGatherMask.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// GatherMask operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GatherMaskOp op)
{
    auto& os = emitter.ostream();
    Value rsvdCntVal = op.getRsvdCnt();
    os << "uint64_t " << emitter.getOrCreateName(rsvdCntVal) << ";\n";

    auto dstType = op.getDst().getType();
    auto src1PatternType = op.getSrc1Pattern().getType();
    if (auto dstLocalTensorType = dyn_cast<ascendc::LocalTensorType>(dstType)) {
        os << ascNamespace << "::" << op.getAPIName() << "<";
        if (failed(emitter.emitType(op.getLoc(), dstLocalTensorType.getElementType())))
            return failure();
        os << ", ";

        if (auto src1LocalTensorType = dyn_cast<ascendc::LocalTensorType>(src1PatternType)) {
            if (failed(emitter.emitType(op.getLoc(), src1LocalTensorType.getElementType())))
                return failure();
            os << ", ";
        }
        os << ascNamespace << "::defaultGatherMaskMode>(" << emitter.getOrCreateName(op.getDst()) << ", "
           << emitter.getOrCreateName(op.getSrc0()) << ", " << emitter.getOrCreateName(op.getSrc1Pattern()) << ", "
           << emitter.getOrCreateName(op.getReduceMode()) << ", " << emitter.getOrCreateName(op.getMask()) << ", "
           << emitter.getOrCreateName(op.getParams()) << ", ";
    } else {
        return op.emitOpError("dst operand must be LocalTensor type");
    }
    os << emitter.getOrCreateName(rsvdCntVal);
    os << ")";

    return success();
}