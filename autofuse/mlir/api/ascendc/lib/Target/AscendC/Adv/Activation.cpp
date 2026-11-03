/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Adv/Activation.h"
using namespace mlir;
using namespace mlir::ascendc;

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SimpleSoftMaxOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    if (failed(emitter.emitType(op.getLoc(), op.getDst().getType().getElementType()))) {
        return failure();
    }
    os << ", " << op.getReuseSource() << ", " << op.getBasicBlock() << ", " << op.getDataFormatNZ() << ">("
       << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSumTensor()) << ", "
       << emitter.getOrCreateName(op.getMaxTensor()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getTiling()) << ", " << emitter.getOrCreateName(op.getSoftmaxShapeInfo()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SoftMaxOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    if (failed(emitter.emitType(op.getLoc(), op.getDst().getType().getElementType()))) {
        return failure();
    }
    os << ", " << op.getReuseSource() << ", " << op.getBasicBlock() << ">(";
    os << emitter.getOrCreateName(op.getDst()) << ", ";
    if (auto sumTensor = op.getSumTensor()) {
        os << emitter.getOrCreateName(sumTensor) << ", ";
    }
    if (auto maxTensor = op.getMaxTensor()) {
        os << emitter.getOrCreateName(maxTensor) << ", ";
    }
    os << emitter.getOrCreateName(op.getSrc()) << ", ";
    if (auto sharedTmpBuffer = op.getSharedTmpBuffer()) {
        os << emitter.getOrCreateName(sharedTmpBuffer) << ", ";
    }
    os << emitter.getOrCreateName(op.getTiling());
    if (auto ssi = op.getSoftmaxShapeInfo()) {
        os << ", " << emitter.getOrCreateName(ssi);
    }
    os << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SoftmaxFlashV2Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    if (failed(emitter.emitType(op.getLoc(), op.getDst().getType().getElementType()))) {
        return failure();
    }
    os << ", " << op.getIsUpdate() << ", " << op.getReuseSource() << ", " << op.getBasicBlock() << ", "
       << op.getDataFormatNZ();
    if (auto config = op.getSoftmaxConfig()) {
        os << ", " << emitter.getOrCreateName(config);
    }
    os << ">(" << emitter.getOrCreateName(op.getDst()) << ", ";
    if (auto outReduceMax = op.getOutReduceMax()) {
        os << emitter.getOrCreateName(outReduceMax) << ", ";
    }
    os << emitter.getOrCreateName(op.getExpSumTensor()) << ", " << emitter.getOrCreateName(op.getMaxTensor()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getExpMaxTensor()) << ", "
       << emitter.getOrCreateName(op.getInExpSumTensor()) << ", " << emitter.getOrCreateName(op.getInMaxTensor())
       << ", ";
    if (auto sharedTmpBuffer = op.getSharedTmpBuffer()) {
        os << emitter.getOrCreateName(sharedTmpBuffer) << ", ";
    }
    os << emitter.getOrCreateName(op.getTiling());
    if (auto shapeInfo = op.getSoftmaxShapeInfo()) {
        os << ", " << emitter.getOrCreateName(shapeInfo);
    }
    os << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SwiGLUOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    if (failed(emitter.emitType(op.getLoc(), op.getDst().getType().getElementType()))) {
        return failure();
    }
    os << ">(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrcTensor0()) << ", "
       << emitter.getOrCreateName(op.getSrcTensor1()) << ", " << emitter.getOrCreateName(op.getScalarValue());
    if (auto stb = op.getSharedTmpBuffer()) {
        os << ", " << emitter.getOrCreateName(stb);
    }
    if (auto cc = op.getCalCount()) {
        os << ", " << emitter.getOrCreateName(cc);
    }
    os << ")";
    return success();
}
