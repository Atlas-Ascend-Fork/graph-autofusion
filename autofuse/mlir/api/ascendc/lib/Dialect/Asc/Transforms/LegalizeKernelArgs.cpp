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
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/EmitAsc/Utils/EmitAscAttributes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_LEGALIZEKERNELARGS
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

BlockArgument appendKernelArgument(func::FuncOp op, emitasc::KernelArgument kind, StringRef name, Type type)
{
    OpBuilder builder(op.getContext());
    NamedAttribute kernelArg(
        builder.getStringAttr(emitasc::attr::kernelArg), builder.getAttr<emitasc::KernelArgumentAttr>(kind));
    unsigned idx = op.getNumArguments();
    if (failed(op.insertArgument(
            idx, type, builder.getDictionaryAttr(kernelArg), NameLoc::get(builder.getStringAttr(name))))) {
        llvm_unreachable("failed to insert kernel argument");
    }
    return op.getArgument(idx);
}

void processKernel(func::FuncOp op, bool setFftsAddr)
{
    auto builder = OpBuilder::atBlockBegin(&op.getFunctionBody().front());
    for (unsigned i = 0; i < op.getNumArguments(); i++) {
        op.setArgAttr(
            i, emitasc::attr::kernelArg,
            builder.getAttr<emitasc::KernelArgumentAttr>(emitasc::KernelArgument::Explicit));
    }
    auto loc = builder.getUnknownLoc();
    if (setFftsAddr) {
        auto as = builder.getI64IntegerAttr(static_cast<int64_t>(ascendc::AddressSpace::gm));
        auto fftsAddr = appendKernelArgument(
            op, emitasc::KernelArgument::FftsAddr, "ffts_addr",
            MemRefType::get(ShapedType::kDynamic, builder.getIntegerType(64, false), AffineMap(), as));
        ascendc::SetFftsBaseAddrOp::create(builder, loc, fftsAddr);
    }
    bool hasMatmul = op.walk([](ascendc::RegistMatmulObjOp) { return WalkResult::interrupt(); }).wasInterrupted();
    bool matmulCubeOnly = op->getParentOfType<ModuleOp>()->hasAttrOfType<UnitAttr>(ascendc::attr::matmulCubeOnly);
    if (hasMatmul && !matmulCubeOnly) {
        Value cond = ascendc::AscendIsAICOp::create(builder, loc, builder.getI1Type());
        auto ifOp = scf::IfOp::create(builder, loc, cond, false);
        builder.setInsertionPointToStart(ifOp.thenBlock());
        Value flag = arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(0xf21));
        ascendc::FftsCrossCoreSyncOp::create(builder, loc, ascendc::Pipe::PIPE_MTE3, flag);
        builder.setInsertionPointAfter(ifOp);
    }
}

struct LegalizeKernelArgsPass : public ascendc::impl::LegalizeKernelArgsBase<LegalizeKernelArgsPass> {
    LegalizeKernelArgsPass(const ascendc::LegalizeKernelArgsOptions& options) : LegalizeKernelArgsBase(options) {}

    void runOnOperation() override
    {
        auto mod = getOperation();
        mod.walk([this](func::FuncOp op) {
            if (op->hasAttrOfType<UnitAttr>(ascendc::attr::global)) {
                processKernel(op, setFftsAddr);
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createLegalizeKernelArgsPass(bool setFftsAddr)
{
    LegalizeKernelArgsOptions options;
    options.setFftsAddr = setFftsAddr;
    return std::make_unique<LegalizeKernelArgsPass>(options);
}
