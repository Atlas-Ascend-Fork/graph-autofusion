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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_HOISTQUEBIND
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

struct HoistQueBindPass : public ascendc::impl::HoistQueBindBase<HoistQueBindPass> {
public:
    void runOnOperation() override
    {
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<
            //
            ascendc::HoistOpPattern<ascendc::QueueOp>, ascendc::HoistOpPattern<ascendc::QueBindOp>,
            ascendc::HoistOpPattern<ascendc::TBufOp>, ascendc::HoistOpPattern<ascendc::TPipeInitBufferOp>,
            ascendc::HoistOpPattern<ascendc::TPipeInitQueueOp>, ascendc::HoistOpPattern<ascendc::TBufGetTensorOp>
            //
            >(context);
        if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed()) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createHoistQueBindPass() { return std::make_unique<HoistQueBindPass>(); }
