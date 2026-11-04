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
#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTQUESYNC
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

Value findQueue(Value tensor)
{
    if (auto op = tensor.getDefiningOp<ascendc::TQueBindAllocTensorOp>()) {
        return op.getQueue();
    }
    if (auto op = tensor.getDefiningOp<ascendc::TQueBindDequeTensorOp>()) {
        return op.getQueue();
    }
    return nullptr;
}

void enqueueTensors(func::FuncOp funcOp)
{
    funcOp.walk([](ascendc::OpWithDst op) {
        auto tensor = op.getDst();
        if (!tensor || !isa<ascendc::BaseTensorType>(tensor.getType()) ||
            isa_and_present<ascendc::GlobalTensorOp>(tensor.getDefiningOp()))
            return;
        OpBuilder builder(op.getContext());
        builder.setInsertionPointAfter(op);
        if (auto queue = findQueue(tensor)) {
            ascendc::TQueBindEnqueTensorOp::create(builder, op.getLoc(), queue, tensor);
            return;
        }
        ascendc::PipeBarrierOp::create(builder, op.getLoc(), ascendc::Pipe::PIPE_V);
    });
}

void createSetGetValueSync(bool isBefore, OpBuilder& builder, Location loc)
{
    ascendc::HardEvent currentEvent = isBefore ? ascendc::HardEvent::V_S : ascendc::HardEvent::S_V;
    Value pipe = ascendc::PipeOp::create(builder, loc);
    auto eventId = ascendc::TPipeFetchEventIDOp::create(builder, loc, builder.getI8Type(), pipe, currentEvent).getResult();
    ascendc::SetFlagOp::create(builder, loc, currentEvent, eventId);
    ascendc::WaitFlagOp::create(builder, loc, currentEvent, eventId);
}

void syncGetValueOp(func::FuncOp& funcOp)
{
    funcOp.walk([](ascendc::APIOp op) {
        if (isa<ascendc::LocalTensorGetValueOp, ascendc::GlobalTensorGetValueOp>(op)) {
            auto loc = op.getLoc();
            OpBuilder builder(op);
            createSetGetValueSync(true, builder, loc);
            builder.setInsertionPointAfter(op);
            createSetGetValueSync(false, builder, loc);
        }
    });
}

void syncSetValueOp(func::FuncOp& funcOp)
{
    funcOp.walk([](ascendc::APIOp op) {
        if (isa<ascendc::LocalTensorSetValueOp, ascendc::GlobalTensorSetValueOp>(op)) {
            auto loc = op.getLoc();
            OpBuilder builder(op);
            if (auto forOp = op->getParentOfType<scf::ForOp>()) {
                constexpr unsigned oneOpOneYield = 2U;
                if (forOp.getBody()->getOperations().size() == oneOpOneYield) {
                    builder.setInsertionPoint(forOp);
                    createSetGetValueSync(true, builder, loc);
                    builder.setInsertionPointAfter(forOp);
                    createSetGetValueSync(false, builder, loc);
                    return;
                }
            }
            createSetGetValueSync(true, builder, loc);
            builder.setInsertionPointAfter(op);
            createSetGetValueSync(false, builder, loc);
        }
    });
}

bool reEnque(OpBuilder& b, Location loc, ascendc::TQueBindEnqueTensorOp enq, ascendc::TQueBindDequeTensorOp deq)
{
    DominanceInfo di;
    if (!di.dominates(enq, deq)) {
        auto* enqParent = deq->getParentRegion()->findAncestorOpInRegion(*enq);
        if (!enqParent) {
            enq.emitOpError("failed to be hoisted to tensor deque op scope");
            return false;
        }
        b.setInsertionPointAfter(enqParent);
        ascendc::TQueBindEnqueTensorOp::create(b, loc, enq.getQueue(), enq.getTensor());
        enq.erase();
    }
    return true;
}

bool dequeueTensors(Region& region)
{
    DominanceInfo di;
    bool res = true;
    for (Block& block : region) {
        for (Operation& op : llvm::make_early_inc_range(block)) {
            auto enq = dyn_cast<ascendc::TQueBindEnqueTensorOp>(op);
            if (!enq) {
                for (Region& inner : op.getRegions()) {
                    res &= dequeueTensors(inner);
                }
                continue;
            }
            auto tensor = enq.getTensor();
            std::vector<Operation*> users;
            llvm::copy_if(tensor.getUsers(), std::back_inserter(users), [&](Operation* user) {
                return !isa<ascendc::TQueBindFreeTensorOp>(user) && ascendc::opPrecedes(enq, user, di);
            });
            if (users.empty()) {
                continue;
            }
            Operation* firstUser = *std::min_element(users.begin(), users.end(), [&](Operation* lhs, Operation* rhs) {
                return ascendc::opPrecedes(lhs, rhs, di);
            });
            auto* userInSameRegion = enq->getParentRegion()->findAncestorOpInRegion(*firstUser);
            if (userInSameRegion) {
                firstUser = userInSameRegion;
            }
            OpBuilder builder(firstUser);
            auto deq = ascendc::TQueBindDequeTensorOp::create(builder, enq.getLoc(), tensor.getType(), enq.getQueue());
            if (!reEnque(builder, op.getLoc(), enq, deq)) {
                return false;
            }
            tensor.replaceUsesWithIf(deq.getTensor(), [&](OpOperand& opnd) {
                auto* owner = opnd.getOwner();
                return llvm::is_contained(users, owner);
            });
        }
    }
    return res;
}

void canonicalizeBarriers(func::FuncOp funcOp)
{
    auto builder = OpBuilder::atBlockTerminator(&funcOp.getFunctionBody().back());
    ascendc::PipeBarrierOp::create(builder, builder.getUnknownLoc(), ascendc::Pipe::PIPE_ALL);
    RewritePatternSet patterns(funcOp.getContext());
    ascendc::PipeBarrierOp::getCanonicalizationPatterns(patterns, funcOp.getContext());
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

struct InsertQueSyncPass : public ascendc::impl::InsertQueSyncBase<InsertQueSyncPass> {
public:
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        enqueueTensors(funcOp);
        if (!dequeueTensors(funcOp.getRegion())) {
            signalPassFailure();
            return;
        }
        syncGetValueOp(funcOp);
        syncSetValueOp(funcOp);
        canonicalizeBarriers(funcOp);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInsertQueSyncPass() { return std::make_unique<InsertQueSyncPass>(); }
