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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_DETECTENABLEDEBUG
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

class DetectEnableDebugPass : public ascendc::impl::DetectEnableDebugBase<DetectEnableDebugPass> {
public:
    void runOnOperation() override
    {
        ModuleOp op = getOperation();
        if (op.walk([](ascendc::PrintfOp) { return WalkResult::interrupt(); }).wasInterrupted()) {
            op->setAttr(ascendc::attr::enableDebug, UnitAttr::get(op->getContext()));
        }
        if (op.walk([](ascendc::DumpTensorOp) { return WalkResult::interrupt(); }).wasInterrupted()) {
            op->setAttr(ascendc::attr::enableDebug, UnitAttr::get(op->getContext()));
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createDetectEnableDebugPass() { return std::make_unique<DetectEnableDebugPass>(); }
