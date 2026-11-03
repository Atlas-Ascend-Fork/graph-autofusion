/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/VecCmpsel.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Compare operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, CompareL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = (emitter.getOrCreateName(op.getDst()) + "_mask_list").str();
    os << "uint64_t " << maskName << "[] = {";
    llvm::interleaveComma(op.getMask(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc0()) << ", " << emitter.getOrCreateName(op.getSrc1()) << ", " << ascNamespace
       << "::CMPMODE::" << ascendc::stringifyEnum(op.getCmpMode()) << ", " << maskName << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, CompareRL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = (emitter.getOrCreateName(op.getSrc0()) + "_mask_list").str();
    os << "uint64_t " << maskName << "[] = {";
    llvm::interleaveComma(op.getMask(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getSrc0()) << ", "
       << emitter.getOrCreateName(op.getSrc1()) << ", " << ascNamespace
       << "::CMPMODE::" << ascendc::stringifyEnum(op.getCmpMode()) << ", " << maskName << ", "
       << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, CompareScalarL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = (emitter.getOrCreateName(op.getDst()) + "_mask_list").str();
    os << "uint64_t " << maskName << "[] = {";
    llvm::interleaveComma(op.getMask(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc0()) << ", " << emitter.getOrCreateName(op.getSrc1Scalar()) << ", "
       << ascNamespace << "::CMPMODE::" << ascendc::stringifyEnum(op.getCmpMode()) << ", " << maskName << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// Select operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, SelectScalarL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSelMask()) << ", " << emitter.getOrCreateName(op.getSrc0()) << ", "
       << emitter.getOrCreateName(op.getSrc1()) << ", " << ascNamespace
       << "::SELMODE::" << ascendc::stringifyEnum(op.getSelMode()) << ", " << maskName << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, SelectL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSelMask()) << ", " << emitter.getOrCreateName(op.getSrc0()) << ", "
       << emitter.getOrCreateName(op.getSrc1()) << ", " << ascNamespace
       << "::SELMODE::" << ascendc::stringifyEnum(op.getSelMode()) << ", " << maskName << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}
