/*
 * Copyright (c) 2025 AISS Group, ISE Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/VecReduce.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// BlockReduceSum operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::BlockReduceSumL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getRepeatTime()) << ", "
       << maskName << ", " << emitter.getOrCreateName(op.getDstRepStride()) << ", "
       << emitter.getOrCreateName(op.getSrcBlkStride()) << ", " << emitter.getOrCreateName(op.getSrcRepStride()) << ")";

    return success();
}

//===----------------------------------------------------------------------===//
// BlockReduceMax operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::BlockReduceMaxL1Op op)
{
    auto& os = emitter.ostream();

    auto maskName = printMask(emitter, op);

    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getRepeatTime()) << ", "
       << maskName << ", " << emitter.getOrCreateName(op.getDstRepStride()) << ", "
       << emitter.getOrCreateName(op.getSrcBlkStride()) << ", " << emitter.getOrCreateName(op.getSrcRepStride()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// BlockReduceMin operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::BlockReduceMinL1Op op)
{
    auto& os = emitter.ostream();

    auto maskName = printMask(emitter, op);

    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getRepeatTime()) << ", "
       << maskName << ", " << emitter.getOrCreateName(op.getDstRepStride()) << ", "
       << emitter.getOrCreateName(op.getSrcBlkStride()) << ", " << emitter.getOrCreateName(op.getSrcRepStride()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// Vector Reduce operations
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// PairReduceSum operations
//===----------------------------------------------------------------------===//
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::PairReduceSumL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getRepeatTime()) << ", "
       << maskName << ", " << emitter.getOrCreateName(op.getDstRepStride()) << ", "
       << emitter.getOrCreateName(op.getSrcBlkStride()) << ", " << emitter.getOrCreateName(op.getSrcRepStride()) << ")";

    return success();
}

//===----------------------------------------------------------------------===//
// ReduceMax operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ReduceMaxL1Op op)
{
    auto& os = emitter.ostream();

    auto maskName = printMask(emitter, op);

    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getSharedTmpBuffer()) << ", "
       << maskName << ", " << emitter.getOrCreateName(op.getRepeatTime()) << ", "
       << emitter.getOrCreateName(op.getSrcRepStride()) << ", " << emitter.getOrCreateName(op.getCalIndex()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// ReduceMin operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ReduceMinL1Op op)
{
    auto& os = emitter.ostream();

    auto maskName = printMask(emitter, op);

    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getSharedTmpBuffer()) << ", "
       << maskName << ", " << emitter.getOrCreateName(op.getRepeatTime()) << ", "
       << emitter.getOrCreateName(op.getSrcRepStride()) << ", " << emitter.getOrCreateName(op.getCalIndex()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// ReduceSum operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ReduceSumL1Op op)
{
    auto& os = emitter.ostream();

    auto maskName = printMask(emitter, op);

    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getSharedTmpBuffer()) << ", "
       << maskName << ", " << emitter.getOrCreateName(op.getRepeatTime()) << ", "
       << emitter.getOrCreateName(op.getSrcRepStride()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// WholeReduceMax/Min operations
//===----------------------------------------------------------------------===//

namespace {
template <typename OpType>
LogicalResult printWholeReduceMaxMinCommon(CodeEmitter& emitter, OpType op)
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << maskName << ", "
       << emitter.getOrCreateName(op.getRepeatTime()) << ", " << emitter.getOrCreateName(op.getDstRepStride()) << ", "
       << emitter.getOrCreateName(op.getSrcBlkStride()) << ", " << emitter.getOrCreateName(op.getSrcRepStride()) << ", "
       << ascNamespace << "::ReduceOrder::" << ascendc::stringifyEnum(op.getOrder()).upper() << ")";
    return success();
}
} // namespace

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::WholeReduceMaxL1Op op)
{
    return printWholeReduceMaxMinCommon(emitter, op);
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::WholeReduceMinL1Op op)
{
    return printWholeReduceMaxMinCommon(emitter, op);
}

//===----------------------------------------------------------------------===//
// WholeReduceSum operations
//===----------------------------------------------------------------------===//
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::WholeReduceSumL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << maskName << ", "
       << emitter.getOrCreateName(op.getRepeatTime()) << ", " << emitter.getOrCreateName(op.getDstRepStride()) << ", "
       << emitter.getOrCreateName(op.getSrcBlkStride()) << ", " << emitter.getOrCreateName(op.getSrcRepStride()) << ")";
    return success();
}
