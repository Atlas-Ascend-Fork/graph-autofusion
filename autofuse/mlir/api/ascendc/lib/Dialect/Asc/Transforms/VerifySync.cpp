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
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Utils/DialectUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_VERIFYSYNC
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

template <typename Transfer>
ascendc::TQueBindAllocTensorOp findDef(TypedValue<ascendc::LocalTensorType> tensor, Transfer& deqToEnq)
{
    if (auto op = tensor.getDefiningOp<ascendc::TQueBindDequeTensorOp>()) {
        return findDef(deqToEnq[op].getTensor(), deqToEnq);
    }
    if (auto op = tensor.getDefiningOp<ascendc::TQueBindAllocTensorOp>()) {
        return op;
    }
    return nullptr;
}

struct VerifySyncPass : public ascendc::impl::VerifySyncBase<VerifySyncPass> {
    void dealTQueBindFreeTensorOp(
        ascendc::TQueBindFreeTensorOp& freeOp, ValueMap<SmallVector<Operation*>>& queBinds,
        std::unordered_map<
            ascendc::TQueBindDequeTensorOp, ascendc::TQueBindEnqueTensorOp,
            PointerLikeTypeHash<ascendc::TQueBindDequeTensorOp>>& deqToEnq)
    {
        auto& operations = queBinds[freeOp.getQueue()];
        auto allocTensorOp = findDef(freeOp.getTensor(), deqToEnq);
        if (allocTensorOp) {
            auto* it = llvm::find_if(operations, [&](Operation* op) {
                auto exAllocOp = dyn_cast<ascendc::TQueBindAllocTensorOp>(op);
                if (!exAllocOp) {
                    return false;
                }
                return exAllocOp.getTensor() == allocTensorOp.getTensor();
            });
            if (it != operations.end()) {
                operations.erase(it);
            } else {
                freeOp.emitWarning()
                    .append(freeOp.getAPIName(), ": tensor memory was freed before its last use")
                    .attachNote(freeOp.getTensor().getLoc())
                    .append("tensor declared here");
            }
        } else {
            freeOp.emitWarning()
                .append(
                    freeOp.getAPIName(), ": there is no corresponding call to ",
                    ascendc::TQueBindAllocTensorOp::getAPIName())
                .attachNote(freeOp.getTensor().getLoc())
                .append("tensor declared here");
        }
    }

    void dealTQueBindDequeTensorOp(
        Operation* oriOp, ascendc::TQueBindDequeTensorOp& deque, ValueMap<SmallVector<Operation*>>& queBinds,
        std::unordered_map<
            ascendc::TQueBindDequeTensorOp, ascendc::TQueBindEnqueTensorOp,
            PointerLikeTypeHash<ascendc::TQueBindDequeTensorOp>>& deqToEnq,
        DominanceInfo& di)
    {
        auto& operations = queBinds[deque.getQueue()];
        auto* it = llvm::find_if(operations, [](Operation* op) { return isa<ascendc::TQueBindEnqueTensorOp>(op); });
        if (it != operations.end()) {
            auto firstEnque = dyn_cast<ascendc::TQueBindEnqueTensorOp>(*it);
            deqToEnq[deque] = firstEnque;
            operations.erase(it);
            auto tensor = firstEnque.getTensor();
            // check that tensor is not used between enque and deque
            std::vector<Operation*> users;
            llvm::copy_if(tensor.getUsers(), std::back_inserter(users), [&](Operation* user) {
                return ascendc::opPrecedes(firstEnque, user, di) && ascendc::opPrecedes(user, deque, di);
            });
            for (auto* op : users) {
                op->emitWarning()
                    .append(
                        "unexpected use of tensor between ", ascendc::TQueBindEnqueTensorOp::getAPIName(), " and ",
                        ascendc::TQueBindDequeTensorOp::getAPIName())
                    .attachNote(tensor.getLoc())
                    .append("tensor declared here");
            }
        } else {
            oriOp->emitWarning()
                .append(
                    deque.getAPIName(), ": there is no corresponding call to ",
                    ascendc::TQueBindEnqueTensorOp::getAPIName())
                .attachNote(deque.getQueue().getLoc())
                .append("queue declared here");
        }
    }

    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        ValueMap<SmallVector<Operation*>> queBinds;
        std::unordered_map<
            ascendc::TQueBindDequeTensorOp, ascendc::TQueBindEnqueTensorOp,
            PointerLikeTypeHash<ascendc::TQueBindDequeTensorOp>>
            deqToEnq;
        DominanceInfo di;
        funcOp.walk([&](Operation* op) {
            if (auto alloc = dyn_cast<ascendc::TQueBindAllocTensorOp>(op)) {
                queBinds[alloc.getQueue()].push_back(op);
            } else if (auto freeOp = dyn_cast<ascendc::TQueBindFreeTensorOp>(op)) {
                dealTQueBindFreeTensorOp(freeOp, queBinds, deqToEnq);
            } else if (auto enque = dyn_cast<ascendc::TQueBindEnqueTensorOp>(op)) {
                queBinds[enque.getQueue()].push_back(op);
            } else if (auto deque = dyn_cast<ascendc::TQueBindDequeTensorOp>(op)) {
                dealTQueBindDequeTensorOp(op, deque, queBinds, deqToEnq, di);
            }
        });
        for (auto& [queBind, operations] : queBinds) {
            if (operations.empty())
                continue;
            for (auto& op : operations) {
                if (auto alloc = dyn_cast<ascendc::TQueBindAllocTensorOp>(op)) {
                    alloc.emitWarning().append(
                        alloc.getAPIName(), ": there is no corresponding call to ",
                        ascendc::TQueBindFreeTensorOp::getAPIName(), " for this tensor");
                } else if (auto enque = dyn_cast<ascendc::TQueBindEnqueTensorOp>(op)) {
                    enque.emitWarning()
                        .append(
                            enque.getAPIName(), ": there is no corresponding call to ",
                            ascendc::TQueBindDequeTensorOp::getAPIName(), " for this tensor")
                        .attachNote(enque.getQueue().getLoc())
                        .append("queue declared here");
                }
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createVerifySyncPass() { return std::make_unique<VerifySyncPass>(); }
