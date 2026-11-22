/*
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/DataConversion.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::ascendc;

namespace {

constexpr std::uint64_t ui64BitWidth = 64;
constexpr size_t transDataBlockSize = 16;

mlir::Type inferElementTypeFromTensorList(ValueRange tensorList)
{
    if (tensorList.empty()) {
        return nullptr;
    }
    mlir::Value firstTensor = tensorList.front();
    if (auto tensorType = dyn_cast<ascendc::LocalTensorType>(firstTensor.getType())) {
        return tensorType.getElementType();
    }
    return nullptr;
}

mlir::Type inferElementTypeFromAddrList(ValueRange addrList)
{
    if (addrList.empty()) {
        return nullptr;
    }
    mlir::Value firstAddr = addrList.front();
    if (!firstAddr.getType().isUnsignedInteger(ui64BitWidth)) {
        return nullptr;
    }

    mlir::Operation* definingOp = firstAddr.getDefiningOp();
    if (definingOp) {
        if (auto getPhyAddrOp = dyn_cast<ascendc::LocalTensorGetPhyAddrOp>(definingOp)) {
            mlir::Value tensorValue = getPhyAddrOp.getTensor();
            if (auto tensorType = dyn_cast<ascendc::LocalTensorType>(tensorValue.getType())) {
                return tensorType.getElementType();
            }
        }
    }
    return nullptr;
}

mlir::Type inferElementTypeFromAddrTensor(mlir::ascendc::TransDataTo5HDOp op)
{
    for (mlir::Value addrTensor : {op.getDst(), op.getSrc()}) {
        for (Operation* user : addrTensor.getUsers()) {
            if (auto setValueOp = dyn_cast<ascendc::LocalTensorSetValueOp>(user)) {
                if (setValueOp.getTensor() != addrTensor) {
                    continue;
                }

                mlir::Value valueToSet = setValueOp.getValue();
                mlir::Operation* definingOp = valueToSet.getDefiningOp();
                if (!definingOp) {
                    continue;
                }

                if (auto getPhyAddrOp = dyn_cast<ascendc::LocalTensorGetPhyAddrOp>(definingOp)) {
                    mlir::Value dataTensor = getPhyAddrOp.getTensor();
                    if (auto tensorType = dyn_cast<ascendc::LocalTensorType>(dataTensor.getType())) {
                        return tensorType.getElementType();
                    }
                }
            }
        }
    }

    return nullptr;
}

} // namespace

//===----------------------------------------------------------------------===//
// Data Conversion operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TransDataTo5HDTensorListOp op)
{
    auto& os = emitter.ostream();
    if (op.getDstList().empty())
        return success();

    mlir::Type elementType = inferElementTypeFromTensorList(op.getDstList());
    if (!elementType) {
        elementType = inferElementTypeFromTensorList(op.getSrcList());
    }
    if (!elementType) {
        return op->emitError("could not infer element type from tensor list");
    }

    auto dstName = (emitter.getOrCreateName(op.getDstList().front()) + "_list").str();
    auto srcName = (emitter.getOrCreateName(op.getSrcList().front()) + "_list").str();
    os << "AscendC::LocalTensor<";
    if (failed(emitter.emitType(op.getLoc(), elementType)))
        return failure();
    os << "> " << dstName << "[] = {";
    llvm::interleaveComma(op.getDstList(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";
    os << "AscendC::LocalTensor<";
    if (failed(emitter.emitType(op.getLoc(), elementType)))
        return failure();
    os << "> " << srcName << "[] = {";
    llvm::interleaveComma(op.getSrcList(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";
    os << ascNamespace << "::" << op.getAPIName() << "<";
    if (failed(emitter.emitType(op.getLoc(), elementType)))
        return failure();
    os << ">(" << dstName << ", " << srcName << ", " << emitter.getOrCreateName(op.getParams()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TransDataTo5HDUintListOp op)
{
    auto& os = emitter.ostream();
    if (op.getDstList().empty())
        return success();

    auto dstName = (emitter.getOrCreateName(op.getParams()) + "_dst_list").str();
    auto srcName = (emitter.getOrCreateName(op.getParams()) + "_src_list").str();
    os << "uint64_t " << dstName << "[] = {";
    llvm::interleaveComma(op.getDstList(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";
    os << "uint64_t " << srcName << "[] = {";
    llvm::interleaveComma(op.getSrcList(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";

    os << ascNamespace << "::" << op.getAPIName() << "<";

    mlir::Type elementType = inferElementTypeFromAddrList(op.getDstList());
    if (!elementType) {
        elementType = inferElementTypeFromAddrList(op.getSrcList());
    }
    if (!elementType) {
        return op->emitError("could not infer element type from tensor list");
    }

    if (failed(emitter.emitType(op.getLoc(), elementType))) {
        return failure();
    }

    os << ">(" << dstName << ", " << srcName << ", " << emitter.getOrCreateName(op.getParams()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TransDataTo5HDTensorOp op)
{
    auto& os = emitter.ostream();
    assert(op.getSrcOffsets().size() == transDataBlockSize && op.getDstOffsets().size() == transDataBlockSize);
    auto srcAddrName = emitter.getOrCreateName(op.getParams()).str() + "_src_addr";
    auto dstAddrName = emitter.getOrCreateName(op.getParams()).str() + "_dst_addr";
    auto dstName = emitter.getOrCreateName(op.getParams()).str() + "_dst_list";
    auto srcName = emitter.getOrCreateName(op.getParams()).str() + "_src_list";

    os << "{\n";
    os.indent();
    os << "uint64_t " << srcAddrName << "=" << emitter.getOrCreateName(op.getSrc()) << ".GetPhyAddr();\n";
    os << "uint64_t " << dstAddrName << "=" << emitter.getOrCreateName(op.getDst()) << ".GetPhyAddr();\n";
    os << "uint64_t " << srcName << "[] = {";
    llvm::interleaveComma(op.getSrcOffsets(), os, [&](int32_t offset) { os << srcAddrName << "+" << offset; });
    os << "};\n";
    os << "uint64_t " << dstName << "[] = {";
    llvm::interleaveComma(op.getDstOffsets(), os, [&](int32_t offset) { os << dstAddrName << "+" << offset; });
    os << "};\n";

    os << ascNamespace << "::" << op.getAPIName() << "<";
    auto elementType = dyn_cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    FAIL_OR(emitter.emitType(op.getLoc(), elementType));
    os << ">(" << dstName << ", " << srcName << ", " << emitter.getOrCreateName(op.getParams()) << ");\n";
    os.unindent();
    os << "}\n";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TransDataTo5HDOp op)
{
    auto& os = emitter.ostream();

    os << ascNamespace << "::" << op.getAPIName() << "<";
    mlir::Type elementType = inferElementTypeFromAddrTensor(op);
    if (!elementType) {
        return op->emitError("could not infer element type from addr tensor");
    }
    if (failed(emitter.emitType(op.getLoc(), elementType))) {
        return failure();
    }

    os << ">(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getParams()) << ")";
    return success();
}
