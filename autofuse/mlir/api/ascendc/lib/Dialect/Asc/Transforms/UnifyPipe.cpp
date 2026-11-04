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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_UNIFYPIPE
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

void unifyPipe(func::FuncOp root)
{
    SmallVector<ascendc::PipeOp> pipes;
    root.walk([&pipes](ascendc::PipeOp op) { pipes.push_back(op); });
    if (pipes.size() == 1) {
        return;
    }
    auto builder = OpBuilder::atBlockBegin(&root.getBody().front());
    Value uniPipe = ascendc::PipeOp::create(builder, builder.getUnknownLoc());
    for (auto pipe : pipes) {
        pipe.replaceAllUsesWith(uniPipe);
        pipe.erase();
    }
}

class UnifyPipePass : public ascendc::impl::UnifyPipeBase<UnifyPipePass> {
    void runOnOperation() override { unifyPipe(getOperation()); }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createUnifyPipePass() { return std::make_unique<UnifyPipePass>(); }
