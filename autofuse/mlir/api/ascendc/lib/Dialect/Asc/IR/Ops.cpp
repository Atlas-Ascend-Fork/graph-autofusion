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
#include "ascir/Dialect/Utils/CVGroupCanonicalization.h"
#include "ascir/Dialect/Utils/DialectUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "ascir/Dialect/Asc/IR/AscendCOps.cpp.inc"

using namespace mlir;
using namespace mlir::ascendc;

namespace {

template <typename CVCondOp>
struct InlineNestedGroup : public OpRewritePattern<CVCondOp> {
    using OpRewritePattern<CVCondOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CVCondOp op, PatternRewriter& rewriter) const override
    {
        if (!op->template getParentOfType<CVCondOp>())
            return failure();
        Block* body = op.getBody();
        auto yieldOp = cast<YieldOp>(body->getTerminator());
        rewriter.inlineBlockBefore(body, op);
        rewriter.replaceOp(op, yieldOp.getOperands());
        rewriter.eraseOp(yieldOp);
        return success();
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// GlobalTensorOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalTensorOp::canonicalize(GlobalTensorOp op, PatternRewriter& rewriter)
{
    return ascir::eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// IfAICOp
//===----------------------------------------------------------------------===//

void IfAICOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context)
{
    results.add<
        ascir::EraseEmptyGroup<IfAICOp, YieldOp>, ascir::EraseUnusedOperands<IfAICOp, YieldOp>,
        ascir::EraseUnusedResults<IfAICOp, YieldOp>, InlineNestedGroup<IfAICOp>>(context);
}

//===----------------------------------------------------------------------===//
// IfAIVOp
//===----------------------------------------------------------------------===//

void IfAIVOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context)
{
    results.add<
        ascir::EraseEmptyGroup<IfAIVOp, YieldOp>, ascir::EraseUnusedOperands<IfAIVOp, YieldOp>,
        ascir::EraseUnusedResults<IfAIVOp, YieldOp>, InlineNestedGroup<IfAIVOp>>(context);
}

//===----------------------------------------------------------------------===//
// GlobalTensorSubIndexOp
//===----------------------------------------------------------------------===//

OpFoldResult GlobalTensorSubIndexOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    if (matchPattern(getIndex(), m_Zero()) && getType() == getTensor().getType())
        return getTensor();
    return {};
}

//===----------------------------------------------------------------------===//
// LocalTensorOp
//===----------------------------------------------------------------------===//

LogicalResult LocalTensorOp::canonicalize(LocalTensorOp op, PatternRewriter& rewriter)
{
    return ascir::eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// LocalTensorSubIndexOp
//===----------------------------------------------------------------------===//

OpFoldResult LocalTensorSubIndexOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    if (matchPattern(getIndex(), m_Zero()) && getType() == getTensor().getType())
        return getTensor();
    return {};
}

//===----------------------------------------------------------------------===//
// PipeBarrierOp
//===----------------------------------------------------------------------===//

LogicalResult PipeBarrierOp::canonicalize(PipeBarrierOp op, PatternRewriter& rewriter)
{
    Block* block = op->getBlock();
    auto nextIt = std::next(Block::iterator(op));
    if (nextIt == block->end())
        return failure();
    if (auto nextOp = dyn_cast<ascendc::PipeBarrierOp>(*nextIt)) {
        if (op.getPipe() == Pipe::PIPE_ALL) {
            rewriter.eraseOp(nextOp);
            return success();
        }
        if (op->getAttrs() == nextOp->getAttrs() || nextOp.getPipe() == Pipe::PIPE_ALL) {
            rewriter.eraseOp(op);
            return success();
        }
    }
    return failure();
}

//===----------------------------------------------------------------------===//
// LocalTensorReinterpretCastOp
//===----------------------------------------------------------------------===//

bool LocalTensorReinterpretCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    return inputs.size() == 1 && outputs.size() == 1 && isa<LocalTensorType>(inputs[0]) &&
           isa<LocalTensorType>(outputs[0]);
}

OpFoldResult LocalTensorReinterpretCastOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    Value in = getIn();
    Type resultType = getResult().getType();
    if (in.getType() == resultType)
        return in;
    if (auto defOp = in.getDefiningOp<LocalTensorReinterpretCastOp>()) {
        Value defIn = defOp.getIn();
        if (resultType == defIn.getType())
            return defIn;
        setOperand(defIn);
        return getResult();
    }
    return {};
}

//===----------------------------------------------------------------------===//
// LocalTensorAutoOp
//===----------------------------------------------------------------------===//

LogicalResult LocalTensorAutoOp::canonicalize(LocalTensorAutoOp op, PatternRewriter& rewriter)
{
    return ascir::eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// RegTensorOp
//===----------------------------------------------------------------------===//

LogicalResult RegTensorOp::canonicalize(RegTensorOp op, PatternRewriter& rewriter)
{
    return ascir::eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// BaseReduce2DL2Op family
//===----------------------------------------------------------------------===//

namespace {
// `srcShape` carries exactly the two extents of the reduce's 2-D view, in the order
// the `layout` attribute names them. Anything else is a malformed op: the emitter
// writes the operand list out as a stack array and the API indexes 0 and 1 out of it
// unconditionally, so a shorter list is an out-of-bounds read at run time and a
// longer one silently drops what does not fit. Neither shows up at build time.
template <typename OpT> LogicalResult verifyReduce2DShapeRank(OpT op)
{
    constexpr size_t kReduceShapeRank = 2;
    if (op.getSrcShape().size() != kReduceShapeRank)
        return op.emitOpError() << "expects exactly " << kReduceShapeRank << " srcShape extents, got "
                                << op.getSrcShape().size();
    return success();
}
} // namespace

LogicalResult ReduceSum2DL2Op::verify()
{
    return verifyReduce2DShapeRank(*this);
}

LogicalResult ReduceSumInt32Op::verify()
{
    auto dst = cast<LocalTensorType>(getDst().getType());
    auto src = cast<LocalTensorType>(getSrc().getType());
    if (!dst.getElementType().isSignlessInteger(32) || src.getElementType() != dst.getElementType()) {
        return emitOpError("expects matching i32 input and destination");
    }
    return verifyReduce2DShapeRank(*this);
}

LogicalResult ReduceMax2DL2Op::verify()
{
    return verifyReduce2DShapeRank(*this);
}

LogicalResult ReduceMin2DL2Op::verify()
{
    return verifyReduce2DShapeRank(*this);
}

LogicalResult ReduceProd2DL2Op::verify()
{
    return verifyReduce2DShapeRank(*this);
}

LogicalResult ReduceAny2DL2Op::verify()
{
    return verifyReduce2DShapeRank(*this);
}

LogicalResult ReduceAll2DL2Op::verify()
{
    return verifyReduce2DShapeRank(*this);
}

//===----------------------------------------------------------------------===//
// AscendCDialect
//===----------------------------------------------------------------------===//

void AscendCDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ascir/Dialect/Asc/IR/AscendCOps.cpp.inc"
        >();
}
