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
#define GEN_PASS_DEF_DETECTKERNELTYPE
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

class DetectKernelTypePass : public ascendc::impl::DetectKernelTypeBase<DetectKernelTypePass> {
public:
    void runOnOperation() override
    {
        auto moduleOp = getOperation();
        if (moduleOp->hasAttr(attr::kernelType))
            return;
        auto hasVectorOps = false;
        auto hasCubeOps = false;
        moduleOp.walk([&hasVectorOps, &hasCubeOps](Operation* op) {
            if (!hasVectorOps && isa<VectorOp>(op))
                hasVectorOps = true;
            if (!hasCubeOps && isa<MmadOp, MmadWithBiasOp>(op))
                hasCubeOps = true;
            if (isa<RegistMatmulObjOp>(op))
                hasVectorOps = hasCubeOps = true;
            if (hasVectorOps && hasCubeOps)
                return WalkResult::interrupt();
            return WalkResult::advance();
        });
        StringRef kernelType;
        if (hasVectorOps && hasCubeOps)
            kernelType = attr::kernelMixed;
        else if (hasCubeOps)
            kernelType = attr::kernelCube;
        else
            kernelType = attr::kernelVector;
        moduleOp->setAttr(attr::kernelType, StringAttr::get(moduleOp.getContext(), kernelType));
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createDetectKernelTypePass() { return std::make_unique<DetectKernelTypePass>(); }
