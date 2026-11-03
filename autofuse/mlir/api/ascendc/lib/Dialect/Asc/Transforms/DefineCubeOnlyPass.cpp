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

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_DEFINECUBEONLY
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

class DefineCubeOnlyPass : public ascendc::impl::DefineCubeOnlyBase<DefineCubeOnlyPass> {
public:
    void runOnOperation() override
    {
        auto mod = getOperation();
        auto builder = ImplicitLocOpBuilder::atBlockBegin(mod->getLoc(), mod.getBody());
        builder.create<emitc::VerbatimOp>("#define ASCENDC_CUBE_ONLY");
        mod->setAttr(ascendc::attr::matmulCubeOnly, builder.getUnitAttr());
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createDefineCubeOnlyPass() { return std::make_unique<DefineCubeOnlyPass>(); }
