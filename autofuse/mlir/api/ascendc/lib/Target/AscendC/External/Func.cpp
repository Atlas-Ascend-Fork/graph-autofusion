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

namespace {

void bindParameterNames(CodeEmitter& codeEmitter, func::FuncOp functionOp)
{
    // Explicit parameter names are part of the external compilation protocol.
    for (BlockArgument arg : functionOp.getArguments()) {
        if (auto name = functionOp.getArgAttrOfType<StringAttr>(arg.getArgNumber(), "ascendc.parameter_name")) {
            codeEmitter.bindParameterName(arg, name.getValue());
        }
    }
}

LogicalResult emitFunctionSignature(CodeEmitter& codeEmitter, func::FuncOp functionOp)
{
    raw_ostream& os = codeEmitter.ostream();
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
    return success();
}

LogicalResult emitBlockDeclarations(CodeEmitter& codeEmitter, func::FuncOp functionOp)
{
    raw_ostream& os = codeEmitter.ostream();
    Region::BlockListType& blocks = functionOp.getBlocks();
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
    return success();
}

LogicalResult emitFunctionBody(CodeEmitter& codeEmitter, func::FuncOp functionOp)
{
    Region::BlockListType& blocks = functionOp.getBlocks();
    for (Block& block : blocks) {
        if (!block.hasNoPredecessors() && failed(codeEmitter.emitLabel(block))) {
            return failure();
        }
        for (Operation& op : block.getOperations()) {
            if (failed(emitOperation(codeEmitter, op, needsSemicolon(op)))) {
                return failure();
            }
        }
    }
    return success();
}

} // namespace

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

    bindParameterNames(codeEmitter, functionOp);
    FAIL_OR(emitFunctionSignature(codeEmitter, functionOp));
    os.indent();
    if (functionOp->hasAttr(ascendc::attr::global)) {
        // 任务类型由桥接层决策后以模块属性 autofuse.kernel_task_type 传入,这里照单打印;
        // 缺省(无属性)兜底 AIV_ONLY,兼容不经桥接的直跑场景。
        // isMainFunction 现在是 emitFunctionSignature 的局部变量,此处用同一判据重算。
        auto module = functionOp->getParentOfType<mlir::ModuleOp>();
        auto taskTypeAttr = module ? module->getAttrOfType<mlir::StringAttr>("autofuse.kernel_task_type")
                                   : mlir::StringAttr();
        os << "KERNEL_TASK_TYPE_DEFAULT("
           << (taskTypeAttr ? taskTypeAttr.getValue() : llvm::StringRef("KERNEL_TYPE_AIV_ONLY"))
           << ");\n";
    }

    Region::BlockListType& blocks = functionOp.getBlocks();
    // Create label names for basic blocks.
    for (Block& block : blocks) {
        codeEmitter.getOrCreateName(block);
    }

    FAIL_OR(emitBlockDeclarations(codeEmitter, functionOp));
    FAIL_OR(emitFunctionBody(codeEmitter, functionOp));

    os.unindent() << "}";

    return success();
}
