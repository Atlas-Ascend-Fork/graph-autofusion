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
#include "ascir/Dialect/Utils/DialectUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_ERASESYNC
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

template <typename OpT>
void eraseOps(Operation* root)
{
    root->walk([](OpT op) { op.erase(); });
}

struct EraseSyncPass : public ascendc::impl::EraseSyncBase<EraseSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        ValueMap<Value> allocTensors; // maps queue to allocated tensor
        funcOp.walk(
            [&allocTensors](ascendc::TQueBindAllocTensorOp op) { allocTensors[op.getQueue()] = op.getTensor(); });
        funcOp.walk([this, &allocTensors](ascendc::TQueBindDequeTensorOp op) {
            auto it = allocTensors.find(op.getQueue());
            if (it == allocTensors.end()) {
                op.emitOpError("doesn't have corresponding alloc_tensor op");
                signalPassFailure();
                return;
            }
            op.replaceAllUsesWith(it->second);
            op.erase();
        });
        eraseOps<ascendc::TQueBindEnqueTensorOp>(funcOp);
        eraseOps<ascendc::SetFlagOp>(funcOp);
        eraseOps<ascendc::WaitFlagOp>(funcOp);
        eraseOps<ascendc::PipeBarrierOp>(funcOp);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createEraseSyncPass() { return std::make_unique<EraseSyncPass>(); }
