/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_UTILS_CVGROUPCANONICALIZATION_H
#define ASCIR_DIALECT_UTILS_CVGROUPCANONICALIZATION_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace ascir {

template <typename CVGroupOp, typename YieldOp>
struct EraseEmptyGroup : public OpRewritePattern<CVGroupOp> {
    using OpRewritePattern<CVGroupOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CVGroupOp op, PatternRewriter& rewriter) const override
    {
        Block* body = op.getBody();
        if (!body->without_terminator().empty())
            return failure();
        auto yieldOperands = cast<YieldOp>(body->getTerminator()).getOperands();
        rewriter.replaceOp(op, yieldOperands);
        return success();
    };
};

template <typename CVGroupOp, typename YieldOp>
struct EraseUnusedOperands : public OpRewritePattern<CVGroupOp> {
    using OpRewritePattern<CVGroupOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CVGroupOp op, PatternRewriter& rewriter) const override
    {
        BitVector unusedOperands(op.getNumOperands());
        Block* body = op.getBody();
        for (unsigned i = 0; i < op.getNumOperands(); i++) {
            auto userInsideGroup = [op](Operation* user) { return op->isProperAncestor(user) && !isa<YieldOp>(user); };
            if (llvm::none_of(op.getOperand(i).getUsers(), userInsideGroup))
                unusedOperands.set(i);
        }
        if (unusedOperands.none())
            return failure();
        rewriter.modifyOpInPlace(op, [&] { op->eraseOperands(unusedOperands); });
        return success();
    };
};

template <typename CVGroupOp, typename YieldOp>
struct EraseUnusedResults : public OpRewritePattern<CVGroupOp> {
    enum struct ResultKind { Used, Forwarded, Unused };

    using OpRewritePattern<CVGroupOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CVGroupOp op, PatternRewriter& rewriter) const override
    {
        unsigned numResults = op.getNumResults();
        if (numResults == 0)
            return failure();
        SmallVector<ResultKind, 4> results(numResults, ResultKind::Used);
        auto yieldOp = cast<YieldOp>(op.getBody()->getTerminator());
        for (unsigned i = 0; i < numResults; i++) {
            auto result = op.getResult(i);
            if (result.use_empty()) {
                results[i] = ResultKind::Unused;
            } else {
                auto* forwardDef = yieldOp.getOperand(i).getDefiningOp();
                if (!forwardDef || !op->isProperAncestor(forwardDef))
                    results[i] = ResultKind::Forwarded;
            }
        }
        if (results.front() == ResultKind::Used && llvm::all_equal(results))
            return failure();
        SmallVector<Type, 4> newTypes;
        for (auto [kind, type] : llvm::zip_equal(results, op.getResultTypes()))
            if (kind == ResultKind::Used)
                newTypes.push_back(type);
        auto newOp = rewriter.create<CVGroupOp>(op.getLoc(), newTypes, op.getOperands());
        rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().end());
        SmallVector<Value, 4> newYields, newResults;
        unsigned resultIdx = 0;
        for (auto [kind, result, yield] : llvm::zip_equal(results, op.getResults(), yieldOp.getOperands())) {
            if (kind == ResultKind::Used) {
                newYields.push_back(yield);
                newResults.push_back(newOp.getResult(resultIdx++));
            } else if (kind == ResultKind::Forwarded) {
                newResults.push_back(yield);
            } else if (kind == ResultKind::Unused) {
                newResults.push_back(Value{});
            }
        }
        rewriter.modifyOpInPlace(yieldOp, [&] { yieldOp->setOperands(newYields); });
        rewriter.replaceOp(op, newResults);
        return success();
    }
};

} // namespace ascir
} // namespace mlir

#endif // ASCIR_DIALECT_UTILS_CVGROUPCANONICALIZATION_H
