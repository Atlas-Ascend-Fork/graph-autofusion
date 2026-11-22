/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/Reg.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Other register API operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::DataCopyLoadOp op)
{
    auto& os = emitter.ostream();
    auto srcTensor = op.getSrc();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), srcTensor.getType().getElementType()));
    os << ", " << ascNamespace << "::Reg::LoadDist::" << ascendc::stringifyEnum(op.getDist()) << ">" << "("
       << emitter.getOrCreateName(op.getDstReg()) << ", " << emitter.getOrCreateName(srcTensor) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::DataCopyStoreOp op)
{
    auto& os = emitter.ostream();
    auto dstTensor = op.getDst();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstTensor.getType().getElementType()));
    os << ", " << ascNamespace << "::Reg::StoreDist::" << ascendc::stringifyEnum(op.getDist()) << ">" << "("
       << emitter.getOrCreateName(dstTensor) << ", " << emitter.getOrCreateName(op.getSrcReg()) << ", "
       << emitter.getOrCreateName(op.getMaskReg()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::UpdateMaskOp op)
{
    auto& os = emitter.ostream();
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    os << " = ";
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << ">(*" << emitter.getOrCreateName(op.getCount()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::RegTensorOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::DuplicateScalarRegOp op)
{
    auto& os = emitter.ostream();
    os << "Duplicate(" << emitter.getOrCreateName(op.getDstReg()) << ", " << emitter.getOrCreateName(op.getScalar())
       << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::LocalMemBarOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<" << ascNamespace
       << "::Reg::MemType::" << ascendc::stringifyEnum(op.getSrc()) << ", " << ascNamespace
       << "::Reg::MemType::" << ascendc::stringifyEnum(op.getDst()) << ">()";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SelectRegOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDstReg()) << ", "
       << emitter.getOrCreateName(op.getSrc0Reg()) << ", " << emitter.getOrCreateName(op.getSrc1Reg()) << ", "
       << emitter.getOrCreateName(op.getMaskReg()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CreateMaskOp op)
{
    auto& os = emitter.ostream();
    auto result = op.getResult();
    FAIL_OR(emitter.emitType(op.getLoc(), result.getType()));
    os << " " << emitter.getOrCreateName(result) << " = " << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getDtype()));
    os << ", " << ascNamespace << "::Reg::MaskPattern::" << ascendc::stringifyEnum(op.getMask()) << ">()";
    return success();
}
