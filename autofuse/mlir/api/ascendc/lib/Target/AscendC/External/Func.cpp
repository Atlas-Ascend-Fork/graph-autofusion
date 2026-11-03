/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/External/Func.h"

using namespace mlir;

LogicalResult mlir::printOperation(CodeEmitter& codeEmitter, func::ConstantOp constantOp)
{
    Operation* operation = constantOp.getOperation();
    Attribute value = constantOp.getValueAttr();

    return printConstantOp(codeEmitter, operation, value);
}

LogicalResult mlir::printOperation(CodeEmitter& codeEmitter, func::CallOp callOp)
{
    if (failed(codeEmitter.emitAssignPrefix(*callOp.getOperation()))) {
        return failure();
    }

    raw_ostream& os = codeEmitter.ostream();
    os << callOp.getCallee() << "(";
    if (failed(codeEmitter.emitOperands(*callOp.getOperation()))) {
        return failure();
    }
    os << ")";
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& codeEmitter, func::ReturnOp returnOp)
{
    raw_ostream& os = codeEmitter.ostream();
    os << "return";
    switch (returnOp.getNumOperands()) {
        case 0:
            return success();
        case 1:
            os << " " << codeEmitter.getOrCreateName(returnOp.getOperand(0));
            return success(codeEmitter.hasValueInScope(returnOp.getOperand(0)));
        default:
            llvm_unreachable("emission for multiple results is not implemented");
    }
}

LogicalResult mlir::printOperation(CodeEmitter& codeEmitter, func::FuncOp functionOp)
{
    // We need to declare variables at top if the function has multiple blocks.
    if (functionOp.getBlocks().size() > 1) {
        return functionOp.emitOpError("with multiple blocks needs variables declared at top");
    }

    CodeEmitter::Scope scope(codeEmitter);
    auto& os = codeEmitter.ostream();

    bool isMainFunction = functionOp->hasAttr(ascendc::attr::global);
    auto args = functionOp.getArguments();

    os << (isMainFunction ? "extern \"C\"  __global__ " : "__inline__ __attribute__((always_inline)) ");
    os << "__aicore__ ";
    FAIL_OR(codeEmitter.emitTypes(functionOp.getLoc(), functionOp.getResultTypes()));
    os << " " << functionOp.getName() << "(";
    FAIL_OR(interleaveCommaWithError(args, os, [&](BlockArgument arg) {
        if (failed(codeEmitter.emitType(functionOp.getLoc(), arg.getType()))) {
            return failure();
        }
        os << " " << codeEmitter.getOrCreateName(arg);
        return success();
    }));
    os << ") {\n";
    os.indent();

    Region::BlockListType& blocks = functionOp.getBlocks();
    // Create label names for basic blocks.
    for (Block& block : blocks) {
        codeEmitter.getOrCreateName(block);
    }

    // Declare variables for basic block arguments.
    for (Block& block : llvm::drop_begin(blocks)) {
        for (BlockArgument& arg : block.getArguments()) {
            if (codeEmitter.hasValueInScope(arg)) {
                return functionOp.emitOpError(" block argument #") << arg.getArgNumber() << " is out of scope";
            }
            if (failed(codeEmitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
                return failure();
            }
            os << " " << codeEmitter.getOrCreateName(arg) << ";\n";
        }
    }

    for (Block& block : blocks) {
        // Only print a label if the block has predecessors.
        if (!block.hasNoPredecessors() && failed(codeEmitter.emitLabel(block))) {
            return failure();
        }
        for (Operation& op : block.getOperations()) {
            if (failed(emitOperation(codeEmitter, op, needsSemicolon(op)))) {
                return failure();
            }
        }
    }

    os.unindent() << "}";

    return success();
}
