/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/AscTransformsPasses.h"
#include "ascir/Dialect/Asc/Utils/Constants.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_MATERIALIZETENSOR
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

struct MaterializeLocalTensor : OpRewritePattern<ascendc::LocalTensorAutoOp> {
    bool alwaysBuf;

    MaterializeLocalTensor(MLIRContext* context, bool alwaysBuf) : alwaysBuf(alwaysBuf), OpRewritePattern(context) {}

    static ascendc::TPosition getPosition(ascendc::LocalTensorAutoOp op)
    {
        if (op.getOutput())
            return ascendc::TPosition::VECOUT;
        if (op.getInput())
            return ascendc::TPosition::VECIN;
        llvm_unreachable("position is undefined because tensor cannot be enqueued");
    }

    LogicalResult matchAndRewrite(ascendc::LocalTensorAutoOp op, PatternRewriter& rewriter) const override
    {
        auto type = op.getType();
        auto loc = op.getLoc();
        ascir::ConstantOpBuilder consts(rewriter);
        Value length;
        auto position = op.getPosition();
        if (type.hasStaticShape()) {
            if (position == ascendc::TPosition::A1 || position == ascendc::TPosition::B1 ||
                position == ascendc::TPosition::A2 || position == ascendc::TPosition::B2 ||
                position == ascendc::TPosition::CO1)
                length = consts.i64(ascendc::getTypeSizeCubeBlockAlign(type, position));
            else
                length = consts.i64(ascendc::getTypeSize(type));
        } else {
            assert(op->getNumOperands() != 0 && "must have operands for dynamic shape");
            auto elementTypeSize = ascendc::getElementTypeSize(type);
            length = consts.i64(elementTypeSize);
            auto align = consts.i64(ascendc::ubBlockSize / elementTypeSize);
            for (auto dim : op.getDynamicShape()) {
                if (position == ascendc::TPosition::A1) {
                    auto ceilDim = arith::CeilDivSIOp::create(rewriter, loc, dim, align);
                    dim = arith::MulIOp::create(rewriter, loc, ceilDim, align);
                }
                length = arith::MulIOp::create(rewriter, loc, length, dim);
            }
        }
        Value pipe = ascendc::PipeOp::create(rewriter, loc);
        if (alwaysBuf || !op.getInput() && !op.getOutput()) {
            auto bufferTy = ascendc::TBufType::get(op.getContext(), op.getPosition());
            Value buffer = ascendc::TBufOp::create(rewriter, loc, bufferTy);
            ascendc::TPipeInitBufferOp::create(rewriter, loc, pipe, buffer, length);
            rewriter.replaceOpWithNewOp<ascendc::TBufGetTensorOp>(op, type, buffer);
            return success();
        }
        auto queueTy = ascendc::QueueType::get(op.getContext(), getPosition(op), 1);
        Value queue = ascendc::QueueOp::create(rewriter, loc, queueTy);
        Value num = consts.i32(1);
        ascendc::TPipeInitQueueOp::create(rewriter, loc, pipe, queue, num, length);
        auto allocOp = rewriter.replaceOpWithNewOp<ascendc::TQueBindAllocTensorOp>(op, type, queue);
        rewriter.setInsertionPoint(allocOp->getBlock()->getTerminator());
        ascendc::TQueBindFreeTensorOp::create(rewriter, loc, queue, allocOp.getTensor());
        return success();
    }
};

struct MaterializeTensorPass : public ascendc::impl::MaterializeTensorBase<MaterializeTensorPass> {
    MaterializeTensorPass(const ascendc::MaterializeTensorOptions& options) : MaterializeTensorBase(options) {}

    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<MaterializeLocalTensor>(context, alwaysBuf);
        if (applyPatternsGreedily(funcOp, std::move(patterns)).failed()) {
            signalPassFailure();
        }
    }
};
} // namespace

std::unique_ptr<Pass> mlir::ascendc::createMaterializeTensorPass(bool alwaysBuf)
{
    MaterializeTensorOptions options;
    options.alwaysBuf = alwaysBuf;
    return std::make_unique<MaterializeTensorPass>(options);
}
