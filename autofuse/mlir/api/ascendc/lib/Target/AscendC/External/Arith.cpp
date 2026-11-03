/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/External/Arith.h"
// Seam: `getResultBitWidth` teaches this emitter about `index`, which is an
// Autofuse requirement; the rule lives with the Autofuse arith emitters.
#include "ascir/Target/Asc/Autofuse/Arith.h"

using namespace mlir;

namespace {
constexpr uint32_t bitWidth32 = 32;
constexpr uint32_t bitWidth16 = 16;
} // namespace

LogicalResult mlir::printOperation(CodeEmitter& emitter, arith::ConstantOp constantOp)
{
    Operation* operation = constantOp.getOperation();
    Attribute value = constantOp.getValue();

    return printConstantOp(emitter, operation, value);
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, arith::MulUIExtendedOp op)
{
    auto& os = emitter.ostream();
    auto resultType = op->getResult(1).getType();
    auto lhs = emitter.getOrCreateName(op.getLhs());
    auto rhs = emitter.getOrCreateName(op.getRhs());
    if (failed(emitter.emitVariableDeclaration(op->getResult(0), false))) {
        return failure();
    }
    os << " = " << lhs << " * " << rhs << ";\n";
    auto typeSize = getResultBitWidth(resultType);
    if (failed(emitter.emitVariableDeclaration(op->getResult(1), false))) {
        return failure();
    }
    if (typeSize == bitWidth32) {
        os << " = (static_cast<uint64_t>(" << lhs << ") * static_cast<uint64_t>(" << rhs << ")) >> 32";
    } else if (typeSize == bitWidth16) {
        os << " = (static_cast<uint32_t>(" << lhs << ") * static_cast<uint32_t>(" << rhs << ")) >> 16";
    } else {
        llvm_unreachable("not implemented");
    }

    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, arith::CmpIOp op)
{
    if (failed(emitter.emitAssignPrefix(*op.getOperation()))) {
        return failure();
    }
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getLhs()) << " ";
    switch (op.getPredicate()) {
        case arith::CmpIPredicate::eq:
            os << "==";
            break;
        case arith::CmpIPredicate::ne:
            os << "!=";
            break;
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::ule:
            os << "<=";
            break;
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::ult:
            os << "<";
            break;
        case arith::CmpIPredicate::sge:
        case arith::CmpIPredicate::uge:
            os << ">=";
            break;
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::ugt:
            os << ">";
            break;
    }
    os << " " << emitter.getOrCreateName(op.getRhs());
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, arith::CmpFOp op)
{
    if (failed(emitter.emitAssignPrefix(*op.getOperation()))) {
        return failure();
    }
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getLhs()) << " ";
    switch (op.getPredicate()) {
        case arith::CmpFPredicate::OEQ:
        case arith::CmpFPredicate::UEQ:
            os << "==";
            break;
        case arith::CmpFPredicate::ONE:
        case arith::CmpFPredicate::UNE:
            os << "!=";
            break;
        case arith::CmpFPredicate::OLE:
        case arith::CmpFPredicate::ULE:
            os << "<=";
            break;
        case arith::CmpFPredicate::OLT:
        case arith::CmpFPredicate::ULT:
            os << "<";
            break;
        case arith::CmpFPredicate::OGE:
        case arith::CmpFPredicate::UGE:
            os << ">=";
            break;
        case arith::CmpFPredicate::OGT:
        case arith::CmpFPredicate::UGT:
            os << ">";
            break;
        case arith::CmpFPredicate::AlwaysFalse:
        case arith::CmpFPredicate::AlwaysTrue:
        case arith::CmpFPredicate::ORD:
        case arith::CmpFPredicate::UNO:
            llvm_unreachable("unsupported predicate in arith.cmpf operation");
    }
    os << " " << emitter.getOrCreateName(op.getRhs());
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, arith::BitcastOp op)
{
    FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
    auto& os = emitter.ostream();
    os << "*reinterpret_cast<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << "*>(&" << emitter.getOrCreateName(op.getIn()) << ")";
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, arith::SelectOp op)
{
    if (failed(emitter.emitAssignPrefix(*op.getOperation()))) {
        return failure();
    }
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getCondition()) << " ? " << emitter.getOrCreateName(op.getTrueValue()) << " : "
       << emitter.getOrCreateName(op.getFalseValue());
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, arith::IndexCastOp op)
{
    if (failed(emitter.emitAssignPrefix(*op.getOperation()))) {
        return failure();
    }
    auto& os = emitter.ostream();
    os << "static_cast<";
    if (failed(emitter.emitType(op.getLoc(), op.getOut().getType()))) {
        return failure();
    }
    os << ">(" << emitter.getOrCreateName(op.getIn()) << ")";
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, arith::NegFOp op)
{
    FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
    auto& os = emitter.ostream();
    os << "-" << emitter.getOrCreateName(op.getOperand());
    return success();
}
