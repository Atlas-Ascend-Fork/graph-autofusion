/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/VecDuplicate.h"
#include "ascir/Target/Asc/Common.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Duplicate operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::DuplicateL0Op op)
{
    auto& os = emitter.ostream();
    FAIL_OR(printIsSetMaskTemplate(emitter, op));
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getScalar()) << ", "
       << emitter.getOrCreateName(op.getMask()) << ", " << emitter.getOrCreateName(op.getRepeatTimes()) << ", "
       << emitter.getOrCreateName(op.getDstBlockStride()) << ", " << emitter.getOrCreateName(op.getDstRepeatStride())
       << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::DuplicateL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);

    FAIL_OR(printIsSetMaskTemplate(emitter, op));
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getScalar()) << ", "
       << maskName << ", " << emitter.getOrCreateName(op.getRepeatTimes()) << ", "
       << emitter.getOrCreateName(op.getDstBlockStride()) << ", " << emitter.getOrCreateName(op.getDstRepeatStride())
       << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::DuplicateL2Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getScalar()) << ", " << emitter.getOrCreateName(op.getCalCount()) << ")";
    return success();
}
