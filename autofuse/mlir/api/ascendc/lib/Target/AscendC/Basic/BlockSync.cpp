/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/BlockSync.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Synchronization operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::PipeBarrierOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<" << ascendc::stringifyEnum(op.getPipe()).upper() << ">()";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::WaitFlagOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::WaitFlag<" << ascNamespace
       << "::HardEvent::" << ascendc::stringifyEnum(op.getEvent()).upper() << ">("
       << emitter.getOrCreateName(op.getEventId()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CrossCoreSetFlagOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::CrossCoreSetFlag<" << static_cast<unsigned>(op.getModeId()) << ", "
       << ascendc::stringifyEnum(op.getPipe()).upper() << ">(" << emitter.getOrCreateName(op.getFlagId()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CrossCoreWaitFlagOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::CrossCoreWaitFlag<" << static_cast<unsigned>(op.getModeId()) << ", "
       << ascendc::stringifyEnum(op.getPipe()).upper() << ">(" << emitter.getOrCreateName(op.getFlagId()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GetBufOp op)
{
    auto& os = emitter.ostream();
    os << op.getAPIName() << "(" << ascendc::stringifyEnum(op.getPipe()).upper() << ", " << op.getBufId() << ", "
       << op.getMode() << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::RlsBufOp op)
{
    auto& os = emitter.ostream();
    os << op.getAPIName() << "(" << ascendc::stringifyEnum(op.getPipe()).upper() << ", " << op.getBufId() << ", "
       << op.getMode() << ")";
    return success();
}
