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
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Constants.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INPUTOUTPUTTENSOR
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

using TensorOp = ascendc::LocalTensorAutoOp;

int getCopyCount(ascendc::BaseTensorType type)
{
    size_t byteSize = llvm::alignTo<ascendc::ubBlockSize>(getTypeSize(type));
    return static_cast<int>(byteSize / ascendc::getElementTypeSize(type));
}

template <typename ControlFlowOp>
void createResultDataCopy(ControlFlowOp op)
{
    for (auto& use : op->getUses()) {
        auto copyOp = dyn_cast<ascendc::DataCopyOp>(use.getOwner());
        if (!copyOp || !copyOp.isLocalToGlobal())
            return;
        OpBuilder builder(op);
        ascir::ConstantOpBuilder consts(builder);
        auto type = cast<ascendc::BaseTensorType>(use.get().getType());
        auto dst = builder.create<TensorOp>(op->getLoc(), type, /*input*/ false, /*output*/ true, ValueRange{});
        builder.setInsertionPointAfter(op);
        Value calCount = consts.i64(getCopyCount(type));
        auto extraOp = builder.create<ascendc::DataCopyL2Op>(op->getLoc(), dst, use.get(), calCount);
        extraOp.setDirection(ascendc::TPosition::VECCALC, ascendc::TPosition::VECCALC);
        copyOp.setSrc(dst);
    }
}

template <typename ControlFlowOp>
void createDataCopyIfNeeded(ControlFlowOp op)
{
    createResultDataCopy(op);
}

template <>
void createDataCopyIfNeeded(scf::ForOp forOp)
{
    createResultDataCopy(forOp);

    auto isUBTensor = [](Value val) {
        auto argPos = ascendc::TPosition::MAX;
        if (auto tensor = val.getDefiningOp<ascendc::LocalTensorAutoOp>())
            argPos = tensor.getPosition();
        return argPos == ascendc::TPosition::VECCALC || argPos == ascendc::TPosition::VECIN ||
               argPos == ascendc::TPosition::VECOUT;
    };

    Liveness liveness(forOp->template getParentOfType<func::FuncOp>());

    const auto& forOpInVals = liveness.getLiveIn(forOp.getBody());
    DenseSet<Value> initVals;
    SmallVector<OpOperand*, 4> operandsForCopy;
    for (auto& operand : forOp.getInitArgsMutable()) {
        Value operandVal = operand.get();
        if (!isUBTensor(operandVal))
            continue;

        // Iter args initialization may be emitted into shallow copy of local tensor objects.
        // Make a deep copy of initialization buffer if it is used as init value multiple times within same forOp,
        // or used in or after the loop to prevent data corruption through writes into iteration argument's buffer.
        if (initVals.contains(operandVal) || forOpInVals.contains(operandVal) ||
            !liveness.isDeadAfter(operandVal, forOp))
            operandsForCopy.push_back(&operand);
        initVals.insert(operandVal);

        // Iter args re-assignment at yield may be emitted into shallow copy of local tensor objects as well.
        // Make a deep copy of yielded arg buffer before yield if corresponding iter arg is used after yielded value
        // definition to prevent data corruption of iter arg buffer through writes into yielded buffer.
        BlockArgument blockIterArg = forOp.getTiedLoopRegionIterArg(&operand);
        OpOperand* yieldOperand = forOp.getTiedLoopYieldedValue(blockIterArg);
        Operation* yieldValDefOp = yieldOperand->get().getDefiningOp();
        if (auto aivOp = dyn_cast_if_present<ascendc::IfAIVOp>(yieldValDefOp)) {
            int resIdx = 0;
            for (auto aivOpRes : aivOp->getResults()) {
                if (aivOpRes == yieldOperand->get())
                    break;
                resIdx++;
            }
            Operation* aivYieldOp = aivOp.getRegion().front().getTerminator();
            yieldOperand = &aivYieldOp->getOpOperand(resIdx);
        }
        if (auto yieldOperandDefOp = ascendc::getAllocationRoot(yieldOperand->get()))
            for (Operation* user : yieldOperand->get().getUsers())
                if (isWriteToAllocation(user, yieldOperandDefOp) && !liveness.isDeadAfter(blockIterArg, user))
                    operandsForCopy.push_back(yieldOperand);
    }

    for (auto* operand : operandsForCopy) {
        Operation* user = operand->getOwner();
        Location loc = user->getLoc();
        OpBuilder builder(user);
        ascir::ConstantOpBuilder consts(builder);
        auto type = cast<ascendc::BaseTensorType>(operand->get().getType());
        Value dst = builder.create<TensorOp>(loc, type, /*input*/ false, /*output*/ false, ValueRange{});
        auto ifAIVOp = builder.create<ascendc::IfAIVOp>(loc, TypeRange{}, ValueRange{});
        builder.setInsertionPointToStart(&ifAIVOp.getRegion().emplaceBlock());
        Value calCount = consts.i64(getCopyCount(type));
        auto copyOp = builder.create<ascendc::DataCopyL2Op>(loc, dst, operand->get(), calCount);
        copyOp.setDirection(ascendc::TPosition::VECCALC, ascendc::TPosition::VECCALC);
        builder.create<ascendc::YieldOp>(loc);
        operand->set(dst);
    }
}

TensorOp findTensorOrigin(Value tensor)
{
    auto* defOp = tensor.getDefiningOp();
    if (!defOp)
        return {};
    if (auto op = dyn_cast<TensorOp>(defOp))
        return op;
    if (auto op = dyn_cast<ascendc::LocalTensorReinterpretCastOp>(defOp))
        return findTensorOrigin(op.getIn());
    if (auto op = dyn_cast<ascendc::LocalTensorSubIndexOp>(defOp))
        return findTensorOrigin(op.getTensor());
    return {};
}

void setInOutTensors(func::FuncOp funcOp)
{
    funcOp.walk([](ascendc::DataCopyOp op) {
        auto src = findTensorOrigin(op.getSrc());
        if (src && op.isLocalToGlobal())
            src.setOutput(true);
        auto dst = findTensorOrigin(op.getDst());
        if (dst && op.isGlobalToLocal())
            dst.setInput(true);
        if (!op.isLocalToLocal())
            return;
        if (isa<ascendc::CopyToL0Op, ascendc::FixpipeOp>(*op)) {
            if (src)
                src.setOutput(true);
            if (dst)
                dst.setInput(true);
        }
    });
    funcOp.walk(createDataCopyIfNeeded<scf::ForOp>);
    funcOp.walk(createDataCopyIfNeeded<scf::IfOp>);
    funcOp.walk(createDataCopyIfNeeded<scf::WhileOp>);
}

void fixInOutTensor(func::FuncOp& funcOp)
{
    funcOp.walk([](TensorOp inTensor) {
        if (!inTensor.getInput() || inTensor.getOutput())
            return;
        auto loc = inTensor.getLoc();
        OpBuilder builder(inTensor);
        auto tensorType = inTensor.getResult().getType();
        inTensor.setOutput(false);
        for (auto& use : inTensor->getUses()) {
            auto* owner = use.getOwner();
            auto copyOp = dyn_cast<ascendc::DataCopyOp>(owner);
            if (!copyOp || !copyOp.isLocalToGlobal())
                return builder.setInsertionPoint(owner);
            ascir::ConstantOpBuilder consts(builder);
            Value calCount = consts.i64(getCopyCount(tensorType));
            auto outTensor = builder.create<TensorOp>(loc, tensorType, /*input*/ false, /*output*/ true, ValueRange{});
            auto extraOp = builder.create<ascendc::DataCopyL2Op>(loc, outTensor, inTensor, calCount);
            extraOp.setDirection(ascendc::TPosition::VECCALC, ascendc::TPosition::VECCALC);
            owner->setOperand(use.getOperandNumber(), outTensor);
        }
    });
}

struct InputOutputTensorPass : public ascendc::impl::InputOutputTensorBase<InputOutputTensorPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        setInOutTensors(funcOp);
        fixInOutTensor(funcOp);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInputOutputTensorPass() { return std::make_unique<InputOutputTensorPass>(); }
