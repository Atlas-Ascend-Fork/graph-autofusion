/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/DialectUtils.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "ascir/Dialect/EmitAsc/IR/EmitAscOps.cpp.inc"

using namespace mlir;
using namespace mlir::emitasc;

//===----------------------------------------------------------------------===//
// InitStructOp
//===----------------------------------------------------------------------===//

void InitStructOp::addField(StringRef name, Value value)
{
    SmallVector<Attribute> names(getFieldNames().getValue());
    names.push_back(StringAttr::get(getContext(), name));
    setFieldNamesAttr(ArrayAttr::get(getContext(), names));
    getFieldValuesMutable().append(value);
}

Value InitStructOp::getField(StringRef name)
{
    auto index = getFieldOperandIndex(name);
    return index ? getFieldValues()[*index] : Value{};
}

std::optional<size_t> InitStructOp::getFieldOperandIndex(StringRef name)
{
    auto names = getFieldNames().getValue();
    const auto* it = llvm::find_if(names, [name](Attribute attr) { return cast<StringAttr>(attr).getValue() == name; });
    if (it == names.end())
        return std::nullopt;
    return std::distance(names.begin(), it);
}

size_t InitStructOp::getNumFields() { return getFieldNames().size(); }

bool InitStructOp::hasField(StringRef name) { return getFieldOperandIndex(name).has_value(); }

ParseResult InitStructOp::parse(OpAsmParser& parser, OperationState& result)
{
    auto& builder = parser.getBuilder();
    Type resultType;
    if (parser.parseType(resultType) || parser.parseLParen())
        return ParseResult::failure();
    result.addTypes(resultType);
    SmallVector<Attribute> names;
    SmallVector<Value> values;
    bool first = true;
    while (parser.parseOptionalRParen().failed()) {
        if (first) {
            first = false;
        } else {
            if (parser.parseComma())
                return ParseResult::failure();
        }
        std::string name;
        OpAsmParser::UnresolvedOperand operand;
        Type type;
        if (parser.parseString(&name) || parser.parseEqual() || parser.parseOperand(operand) ||
            parser.parseColonType(type) || parser.resolveOperand(operand, type, values))
            return ParseResult::failure();
        names.push_back(builder.getStringAttr(name));
    }
    result.addAttribute(getAttributeNames()[0], builder.getArrayAttr(names));
    result.addOperands(values);
    return ParseResult::success();
}

void InitStructOp::print(OpAsmPrinter& printer)
{
    printer << ' ' << getType() << '(';
    bool first = true;
    for (auto [name, value] : llvm::zip_equal(getFieldNames(), getFieldValues())) {
        if (first)
            first = false;
        else
            printer << ", ";
        printer << cast<StringAttr>(name) << " = " << value << " : " << value.getType();
    }
    printer << ')';
}

void InitStructOp::setField(StringRef name, Value value)
{
    auto index = getFieldOperandIndex(name);
    if (!index)
        return;
    getFieldValuesMutable().slice(*index, 1).assign(value);
}

LogicalResult InitStructOp::verify()
{
    if (getFieldNames().size() != getFieldValues().size())
        return emitOpError("must have number of field names equal to number of field values");
    return success();
}

//===----------------------------------------------------------------------===//
// PtrOffsetOp
//===----------------------------------------------------------------------===//

Value PtrOffsetOp::getViewSource() { return getBase(); }

OpFoldResult PtrOffsetOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    if (auto offset = getStaticOffset())
        return offset->isZero() ? getBase() : nullptr;
    if (auto offset = getDynamicOffset())
        return isConstantIntValue(offset, 0) ? getBase() : nullptr;
    return nullptr;
}

//===----------------------------------------------------------------------===//
// ReinterpretCastOp
//===----------------------------------------------------------------------===//

bool ReinterpretCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    return inputs.size() == 1U && outputs.size() == 1U;
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

bool VariableOp::isStatic() { return getStaticInit().has_value(); }

OpFoldResult VariableOp::getInit(bool fold)
{
    auto dynamicInit = getDynamicInit();
    if (dynamicInit)
        return fold ? getAsOpFoldResult(dynamicInit) : dynamicInit;
    auto staticInit = getStaticInit();
    assert(staticInit.has_value() && "either static or dynamic init must exist");
    return staticInit.value();
}

LogicalResult VariableOp::canonicalize(VariableOp op, PatternRewriter& rewriter)
{
    return ascir::eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// VFForOp
//===----------------------------------------------------------------------===//

void VFForOp::build(OpBuilder& builder, OperationState& state, Value upperBound)
{
    OpBuilder::InsertionGuard guard(builder);
    state.addOperands(upperBound);
    Type type = builder.getIndexType();
    Region* bodyRegion = state.addRegion();
    Block* bodyBlock = builder.createBlock(bodyRegion);
    bodyBlock->addArgument(type, state.location);
    ensureTerminator(*bodyRegion, builder, state.location);
}

LogicalResult VFForOp::canonicalize(VFForOp op, PatternRewriter& rewriter)
{
    Block& block = op.getRegion().front();
    if (block.without_terminator().empty()) {
        rewriter.eraseOp(op);
        return success();
    }
    if (auto ub = getConstantIntValue(op.getUpperBound()); ub && ub.value() == 0) {
        rewriter.eraseOp(op);
        return success();
    }
    return failure();
}

LogicalResult VFForOp::verify()
{
    if (getBody()->getArguments().size() != 1)
        return emitOpError("block must have one argument");
    if (!getOperation()->getParentOfType<emitasc::VecScopeOp>()) {
        return emitOpError("must be inside emitasc.vec_scope block");
    }
    return success();
}

SmallVector<Region*> VFForOp::getLoopRegions()
{
    SmallVector<Region*> regions;
    regions.push_back(&getRegion());
    return regions;
}

//===----------------------------------------------------------------------===//
// EmitAscDialect
//===----------------------------------------------------------------------===//

void EmitAscDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ascir/Dialect/EmitAsc/IR/EmitAscOps.cpp.inc"
        >();
}
