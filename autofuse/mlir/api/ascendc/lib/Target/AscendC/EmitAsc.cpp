/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/EmitAscTarget.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/External/Scf.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::emitasc;

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::CallOpaqueOp op)
{
    auto& os = emitter.ostream();

    FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
    os << op.getCallee() << '(';
    llvm::interleaveComma(op.getOperands(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << ')';

    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::CopyStructOp op)
{
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    auto base = emitter.getOrCreateName(op.getBase());
    auto result = emitter.getOrCreateName(op.getResult());
    auto& os = emitter.ostream();
    os << ' ' << result << ";\n";
    os << "for (size_t i = 0; i < sizeof(" << result << "); i++) {\n";
    os.indent() << "auto byte = reinterpret_cast<";
    if (auto attr = dyn_cast_if_present<IntegerAttr>(op.getBase().getType().getMemorySpace())) {
        auto asInt = static_cast<uint8_t>(attr.getValue().getSExtValue());
        if (auto addrSpace = ascendc::symbolizeAddressSpace(asInt)) {
            emitter.emitAddressSpace(*addrSpace);
        }
    }
    os << "uint8_t*>(" << base << ")[i];\n";
    os << "reinterpret_cast<uint8_t*>(&" << result << ")[i] = byte;\n";
    os.unindent() << '}';
    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::DereferenceOp op)
{
    auto& os = emitter.ostream();
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << "& " << emitter.getOrCreateName(op.getResult()) << " = *" << emitter.getOrCreateName(op.getBase());
    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::MaskOp op)
{
    // Mask in AscendC: [maskL, maskH]
    auto& os = emitter.ostream();
    os << "uint64_t " << emitter.getOrCreateName(op.getResult()) << "[2] = {static_cast<uint64_t>("
       << emitter.getOrCreateName(op.getMaskL()) << "), static_cast<uint64_t>("
       << emitter.getOrCreateName(op.getMaskH()) << ")}";
    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::MemberOp op)
{
    auto& os = emitter.ostream();
    FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
    os << emitter.getOrCreateName(op.getBase());
    if (isa<MemRefType>(op.getBase().getType()))
        os << "->";
    else
        os << '.';
    os << op.getField();
    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::MemberPtrOp op)
{
    auto& os = emitter.ostream();

    FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
    os << "reinterpret_cast<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << ">(" << '&' << emitter.getOrCreateName(op.getBase()) << "->";
    if (auto attr = op.getFieldAttr()) {
        os << attr.str();
    } else {
        os << emitter.structFieldNamePrefix << op.getIndex();
    }
    os << ')';

    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::DeclarePyStructOp op)
{
    auto& os = emitter.ostream();
    os << "#pragma pack(push, 8)\n";
    auto pType = dyn_cast<emitasc::PyStructType>(op.getPystruct());
    os << "struct " << pType.getNameAttr().getValue() << " {\n";
    os.indent();
    for (auto [typeAttr, nameAttr] : llvm::zip_equal(pType.getTypesAttr(), pType.getNamesAttr())) {
        FAIL_OR(emitter.emitType(op.getLoc(), cast<TypeAttr>(typeAttr).getValue()));
        os << " " << cast<StringAttr>(nameAttr).getValue() << ";\n";
    }
    os.unindent() << "};\n";
    os << "#pragma pack(pop)\n";

    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::InitStructOp op)
{
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    auto& os = emitter.ostream();
    auto result = emitter.getOrCreateName(op.getResult());
    os << ' ' << result;
    for (auto [name, value] : llvm::zip_equal(op.getFieldNames(), op.getFieldValues()))
        os << ";\n" << result << '.' << cast<StringAttr>(name).getValue() << " = " << emitter.getOrCreateName(value);
    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::MemberRefOp op)
{
    auto& os = emitter.ostream();

    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << "& " << emitter.getOrCreateName(op.getResult()) << " = reinterpret_cast<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << "&>(" << emitter.getOrCreateName(op.getBase()) << "->";
    if (auto attr = op.getFieldAttr()) {
        os << attr.str();
    } else {
        os << emitter.structFieldNamePrefix << op.getIndex();
    }
    os << ')';

    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::PtrOffsetOp op)
{
    FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getBase()) << " + ";
    if (auto offset = op.getDynamicOffset()) {
        os << emitter.getOrCreateName(offset);
    } else {
        FAIL_OR(emitter.emitAttribute(op.getLoc(), op.getStaticOffsetAttr()));
    }
    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::ReinterpretCastOp op)
{
    auto& os = emitter.ostream();

    FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
    os << "reinterpret_cast<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << ">(" << emitter.getOrCreateName(op.getSource()) << ')';

    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::SetMemberOp op)
{
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getBase()) << "." << op.getField() << " = "
       << emitter.getOrCreateName(op.getValue());

    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::VariableOp op)
{
    auto& os = emitter.ostream();
    auto loc = op.getLoc();
    auto res = op.getResult();
    auto resType = res.getType();

    FAIL_OR(emitter.emitType(op.getLoc(), resType.getElementType()));
    os << ' ' << emitter.getOrCreateName(res);

    for (auto size : resType.getShape()) {
        os << '[' << size << ']';
    }

    os << '{';
    if (op.isStatic()) {
        FAIL_OR(emitter.emitAttribute(loc, op.getStaticInitAttr()));
    } else {
        os << emitter.getOrCreateName(op.getDynamicInit());
    }
    os << '}';

    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::VerbatimOp op)
{
    auto& os = emitter.ostream();
    auto args = op.getArgs();
    auto code = op.getValue();
    if (args.empty()) {
        os << code;
        return success();
    }
    std::string result;
    result.reserve(2 * code.size()); // the factor of 2 is used to ensure sufficient space.
    size_t i = 1;
    size_t rem = 0;
    const char* data = code.data();
    while (i < code.size()) {
        if (code[i - 1] != '$') {
            i++;
            continue;
        }
        size_t j = i;
        while (j < code.size() && isdigit(code[j])) {
            j++;
        }
        if (j - i > 0) {
            size_t index = 0;
            auto fcResult = std::from_chars(data + i, data + j, index);
            if (!std::make_error_code(fcResult.ec) && index < args.size()) {
                std::copy(data + rem, data + i - 1, std::back_inserter(result));
                result += emitter.getOrCreateName(args[index]);
                rem = j;
                i = j;
                continue;
            }
        }
        i++;
    }
    std::copy(data + rem, data + code.size(), std::back_inserter(result));
    os << result;
    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::VFForOp op)
{
    auto& os = emitter.ostream();
    os << "for (";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getUnderlyingType()));
    os << " " << emitter.getOrCreateName(op.getInductionVar());
    os << " = " << op.getLowerBoundAsInt();
    os << "; " << emitter.getOrCreateName(op.getInductionVar());
    os << " < static_cast<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getUnderlyingType()));
    os << ">(" << emitter.getOrCreateName(op.getUpperBound()) << ")";
    os << "; " << emitter.getOrCreateName(op.getInductionVar());
    os << " += " << op.getStepAsInt() << ") {\n";
    os.indent();
    Region& forRegion = op.getRegion();
    auto regionOps = forRegion.getOps();
    for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
        Operation& op = *it;
        if (failed(emitOperation(emitter, op, needsSemicolon(op)))) {
            return failure();
        }
    }
    os.unindent() << "}";
    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter& emitter, emitasc::VecScopeOp op)
{
    auto& os = emitter.ostream();
    os << "__VEC_SCOPE__\n";
    os << "{\n";
    os.indent();
    FAIL_OR(emitBlock(emitter, *op.getBody()));
    os.unindent() << "}";
    return success();
}

LogicalResult mlir::emitasc::printOperation(CodeEmitter&, emitasc::YieldOp)
{
    // No need to print anything
    return success();
}
