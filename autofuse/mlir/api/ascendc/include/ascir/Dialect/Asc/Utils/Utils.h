/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASC_UTILS_UTILS_H
#define ASCIR_DIALECT_ASC_UTILS_UTILS_H

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace ascendc {

template <typename OpT>
struct HoistOpPattern : public OpRewritePattern<OpT> {
    using OpRewritePattern<OpT>::OpRewritePattern;

    virtual bool hoistable(OpT) const { return true; }

    LogicalResult matchAndRewrite(OpT op, PatternRewriter& rewriter) const override
    {
        Operation* parent = op->getParentOp();
        if (isa<func::FuncOp>(parent))
            return failure();
        if (!hoistable(op))
            return failure();
        DominanceInfo di;
        bool dominatedByOperands =
            llvm::all_of(op->getOperands(), [&](Value opnd) { return di.dominates(opnd, parent); });
        if (!dominatedByOperands)
            return failure();
        rewriter.setInsertionPoint(parent);
        rewriter.replaceOp(op, rewriter.clone(*op.getOperation())->getResults());
        return success();
    }
};

int64_t getTypeSize(Type type);

int64_t getTypeSizeCubeBlockAlign(ShapedType type, TPosition position);

int64_t getElementTypeSize(ShapedType type);

bool opPrecedes(Operation* lhs, Operation* rhs);

bool opPrecedes(Operation* lhs, Operation* rhs, DominanceInfo& di);

void registerInlinerInterfaces(DialectRegistry& registry);

ModuleOp getModule(Operation* op);

StringRef getCompilationArch(Operation* op);

StringRef getSocVersion(Operation* op);

std::optional<int64_t> getVecLen(Operation* op);

bool isTargetArchC310(Operation* op);

SmallVector<Operation*> collectAllUsers(ascendc::LocalTensorAutoOp tensorOp);

ascendc::LocalTensorAutoOp getAllocationRoot(Value v);

Pipe getOpPipe(Operation* op, Pipe defaultPipe = Pipe::PIPE_S);

bool isWriteToAllocation(Operation* op, ascendc::LocalTensorAutoOp root);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_DIALECT_ASC_UTILS_UTILS_H
