/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/DataCopy.h"

using namespace mlir;
using namespace mlir::ascendc;

namespace {

template <typename CopyOpTy>
LogicalResult emitCopyTemplateArgs(CodeEmitter& emitter, CopyOpTy op)
{
    auto& os = emitter.ostream();

    os << "<";
    auto tensorType = cast<LocalTensorType>(op.getDst().getType());
    FAIL_OR(emitter.emitType(op.getLoc(), tensorType.getElementType()));
    os << ", ";
    os << emitter.getOrCreateName(op.getIsSetMask());
    os << ">";

    return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Data transfer operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::DataCopySliceOp op)
{
    auto& os = emitter.ostream();
    auto dstName = (emitter.getOrCreateName(op.getDst()) + "_slice_info").str();
    auto srcName = (emitter.getOrCreateName(op.getSrc()) + "_slice_info").str();

    os << "AscendC::SliceInfo " << dstName << "[] = {";
    llvm::interleaveComma(op.getDstSliceInfo(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";
    os << "AscendC::SliceInfo " << srcName << "[] = {";
    llvm::interleaveComma(op.getSrcSliceInfo(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";

    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << dstName << ", " << srcName << ", "
       << emitter.getOrCreateName(op.getDimValue()) << ')';

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CopyL0Op op)
{
    auto& os = emitter.ostream();

    auto maskName = printMask(emitter, op);

    os << ascNamespace << "::" << op.getAPIName();
    FAIL_OR(emitCopyTemplateArgs(emitter, op));

    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << maskName << ", " << emitter.getOrCreateName(op.getRepeatTime()) << ", "
       << emitter.getOrCreateName(op.getRepeatParams()) << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CopyL1Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName();
    FAIL_OR(emitCopyTemplateArgs(emitter, op));

    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getMask()) << ", " << emitter.getOrCreateName(op.getRepeatTime()) << ", "
       << emitter.getOrCreateName(op.getRepeatParams()) << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::NdDmaParamsOp op)
{
    auto& os = emitter.ostream();
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    os << "=";
    os << mlir::ascNamespace << "::NdDmaParams<";
    auto type = op.getPadValue().getType();
    FAIL_OR(emitter.emitType(op.getLoc(), type));
    os << "," << op.getDim() << ">";
    os << "{{{";
    for (const auto& val : op.getSrcStride()) {
        os << "static_cast<uint64_t>(" << emitter.getOrCreateName(val) << "),";
    }
    os << "},{";
    llvm::interleaveComma(
        op.getDstStride(), os, [&](Attribute attr) { (void)emitter.emitAttribute(op.getLoc(), attr); });
    os << "},{";
    for (const auto& val : op.getSize()) {
        os << "static_cast<uint32_t>(" << emitter.getOrCreateName(val) << "),";
    }
    os << "},{";
    llvm::interleaveComma(op.getPadLeft(), os, [&](Attribute attr) { (void)emitter.emitAttribute(op.getLoc(), attr); });
    os << "},{";
    llvm::interleaveComma(
        op.getPadRight(), os, [&](Attribute attr) { (void)emitter.emitAttribute(op.getLoc(), attr); });
    os << "}},";
    os << emitter.getOrCreateName(op.getPadValue()) << "}";
    return success();
}
