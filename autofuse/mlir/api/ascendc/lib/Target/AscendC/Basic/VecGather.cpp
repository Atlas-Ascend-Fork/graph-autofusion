/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/VecGather.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Gatherb operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GatherbL0Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc0()) << ", " << emitter.getOrCreateName(op.getOffset()) << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// Gather operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GatherL0Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getSrcOffset()) << ", "
       << emitter.getOrCreateName(op.getSrcBaseAddr()) << ", " << emitter.getOrCreateName(op.getMask()) << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getDstRepStride()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GatherL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);

    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getDstOffset()) << ", "
       << emitter.getOrCreateName(op.getDstBase()) << ", " << maskName << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getSrcRepStride()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GatherL2Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getSrcOffset()) << ", "
       << emitter.getOrCreateName(op.getSrcBaseAddr()) << ", " << emitter.getOrCreateName(op.getCount()) << ")";
    return success();
}
