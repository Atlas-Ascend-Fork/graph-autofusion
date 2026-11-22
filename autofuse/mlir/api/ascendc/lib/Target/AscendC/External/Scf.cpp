/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/External/Scf.h"

using namespace mlir;

LogicalResult mlir::emitBlock(CodeEmitter& codeEmitter, Block& block)
{
    for (auto& op : block) {
        if (isa<scf::YieldOp>(op) && op.getNumOperands() == 0) {
            continue;
        }
        FAIL_OR(emitOperation(codeEmitter, op, needsSemicolon(op)));
    }
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& codeEmitter, scf::ForOp forOp)
{
    raw_indented_ostream& os = codeEmitter.ostream();

    OperandRange operands = forOp.getInitArgs();
    Block::BlockArgListType iterArgs = forOp.getRegionIterArgs();
    Operation::result_range results = forOp.getResults();

    for (OpResult result : results) {
        if (failed(codeEmitter.emitVariableDeclaration(result, /*trailingSemicolon=*/true))) {
            return failure();
        }
    }

    for (auto pair : llvm::zip(iterArgs, operands)) {
        if (failed(codeEmitter.emitType(forOp.getLoc(), std::get<0>(pair).getType()))) {
            return failure();
        }
        os << " " << codeEmitter.getOrCreateName(std::get<0>(pair)) << " = ";
        os << codeEmitter.getOrCreateName(std::get<1>(pair)) << ";";
        os << "\n";
    }

    os << "for (";
    FAIL_OR(codeEmitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType()));
    os << " " << codeEmitter.getOrCreateName(forOp.getInductionVar());
    os << " = " << codeEmitter.getOrCreateName(forOp.getLowerBound());
    os << "; " << codeEmitter.getOrCreateName(forOp.getInductionVar());
    os << " < " << codeEmitter.getOrCreateName(forOp.getUpperBound());
    os << "; " << codeEmitter.getOrCreateName(forOp.getInductionVar());
    os << " += " << codeEmitter.getOrCreateName(forOp.getStep()) << ") {\n";
    os.indent();

    Region& forRegion = forOp.getRegion();
    auto regionOps = forRegion.getOps();

    // We skip the trailing yield op because this updates the result variables
    // of the for op in the generated code. Instead we update the iterArgs at
    // the end of a loop iteration and set the result variables after the for
    // loop.
    for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
        Operation& op = *it;
        if (failed(emitOperation(codeEmitter, op, needsSemicolon(op)))) {
            return failure();
        }
    }

    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    // Copy yield operands into iterArgs at the end of a loop iteration.
    for (auto [iterArg, operand] : llvm::zip(iterArgs, yieldOp.getResults())) {
        os << codeEmitter.getOrCreateName(iterArg) << " = " << codeEmitter.getOrCreateName(operand) << ";\n";
    }

    os.unindent() << "}";

    if (!results.empty()) {
        os << "\n";
    }
    // Copy iterArgs into results after the for loop.
    llvm::interleave(
        llvm::zip(results, iterArgs),
        [&](auto pair) {
            auto& [result, iterArg] = pair;
            os << codeEmitter.getOrCreateName(result) << " = " << codeEmitter.getOrCreateName(iterArg) << ";";
        },
        [&] { os << "\n"; });
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& codeEmitter, scf::IfOp ifOp)
{
    auto& os = codeEmitter.ostream();

    for (OpResult result : ifOp.getResults()) {
        if (failed(codeEmitter.emitVariableDeclaration(
                result,
                /*trailingSemicolon=*/true))) {
            return failure();
        }
    }
    os << "if (";
    if (failed(codeEmitter.emitOperands(*ifOp.getOperation()))) {
        return failure();
    }
    os << ") {\n";
    os.indent();
    FAIL_OR(emitBlock(codeEmitter, *ifOp.thenBlock()));
    os.unindent() << "}";
    if (!ifOp.getElseRegion().empty()) {
        os << " else {\n";
        os.indent();
        FAIL_OR(emitBlock(codeEmitter, *ifOp.elseBlock()));
        os.unindent() << "}";
    }
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& codeEmitter, scf::IndexSwitchOp op)
{
    for (auto result : op.getResults()) {
        FAIL_OR(codeEmitter.emitVariableDeclaration(result, true));
    }
    auto& os = codeEmitter.ostream();
    os << "switch(" << codeEmitter.getOrCreateName(op.getArg()) << ") {\n";
    for (auto [i, value] : llvm::enumerate(op.getCases())) {
        os << "case " << value << ": {\n";
        os.indent();
        FAIL_OR(emitBlock(codeEmitter, op.getCaseBlock(i)));
        os.unindent() << "} break;\n";
    }
    os << "default: {\n";
    os.indent();
    FAIL_OR(emitBlock(codeEmitter, op.getDefaultBlock()));
    os.unindent() << "}\n}";
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& codeEmitter, scf::YieldOp yieldOp)
{
    auto& os = codeEmitter.ostream();
    Operation* parentOp = yieldOp->getParentOp();
    return interleaveWithError(
        llvm::zip(parentOp->getResults(), yieldOp.getOperands()),
        [&](auto pair) -> LogicalResult {
            auto [result, operand] = pair;
            if (!codeEmitter.hasValueInScope(operand))
                return yieldOp.emitError("operand value not in scope");
            os << codeEmitter.getOrCreateName(result) << " = " << codeEmitter.getOrCreateName(operand) << ";";
            return success();
        },
        [&] { os << "\n"; });
}

LogicalResult mlir::printOperation(CodeEmitter& codeEmitter, scf::ConditionOp conditionOp)
{
    raw_indented_ostream& os = codeEmitter.ostream();
    os << "if (!" << codeEmitter.getOrCreateName(conditionOp.getCondition()) << ") {\n";
    os.indent();
    Operation& parentOp = *conditionOp.getOperation()->getParentOp();
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
        for (auto [result, arg] : llvm::zip(whileOp.getResults(), conditionOp.getArgs())) {
            os << codeEmitter.getOrCreateName(result) << " = " << codeEmitter.getOrCreateName(arg) << ";\n";
        }
    }
    os << "break;\n";
    os.unindent();
    os << "}";
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& codeEmitter, scf::WhileOp whileOp)
{
    auto& os = codeEmitter.ostream();
    for (OpResult result : whileOp.getResults()) {
        FAIL_OR(codeEmitter.emitVariableDeclaration(result, true));
    }
    auto beforeArgs = whileOp.getBeforeArguments();
    for (auto [arg, init] : llvm::zip(beforeArgs, whileOp.getInits())) {
        FAIL_OR(codeEmitter.emitType(whileOp.getLoc(), arg.getType()));
        os << " " << codeEmitter.getOrCreateName(arg) << " = " << codeEmitter.getOrCreateName(init) << ";\n";
    }
    os << "while (true) {\n";
    os.indent();
    for (Operation& op : whileOp.getBefore().getOps()) {
        FAIL_OR(emitOperation(codeEmitter, op, /*trailingSemicolon=*/true));
    }
    auto afterArgs = whileOp.getAfterArguments();
    auto conditionOpArgs = whileOp.getConditionOp().getArgs();
    for (auto [arg, init] : llvm::zip(afterArgs, conditionOpArgs)) {
        FAIL_OR(codeEmitter.emitType(whileOp.getLoc(), arg.getType()));
        os << " " << codeEmitter.getOrCreateName(arg) << " = " << codeEmitter.getOrCreateName(init) << ";\n";
    }
    for (Operation& op : whileOp.getAfter().getOps()) {
        if (auto yield = dyn_cast<scf::YieldOp>(op)) {
            for (auto [result, operand] : llvm::zip(beforeArgs, yield.getOperands()))
                os << codeEmitter.getOrCreateName(result) << " = " << codeEmitter.getOrCreateName(operand) << ";\n";
            continue;
        }
        FAIL_OR(emitOperation(codeEmitter, op, /*trailingSemicolon=*/true));
    }
    os.unindent() << "}";
    return success();
}
