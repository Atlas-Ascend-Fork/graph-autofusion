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

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_GENERATEBOILERPLATE
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

class GenerateBoilerplatePass : public ascendc::impl::GenerateBoilerplateBase<GenerateBoilerplatePass> {
public:
    void runOnOperation() override
    {
        auto mod = getOperation();
        auto builder = ImplicitLocOpBuilder::atBlockBegin(mod->getLoc(), mod.getBody());
        builder.create<emitc::IncludeOp>("kernel_operator.h");
        bool hasListTensorDesc =
            mod.walk([](ascendc::ListTensorDescOp) { return WalkResult::interrupt(); }).wasInterrupted();
        bool hasListTensorDescV2 =
            mod.walk([](ascendc::ListTensorDescV2Op) { return WalkResult::interrupt(); }).wasInterrupted();
        if (hasListTensorDesc || hasListTensorDescV2) {
            builder.create<emitc::IncludeOp>("kernel_operator_list_tensor_intf.h");
        }
        bool hasMatmul = mod.walk([](ascendc::RegistMatmulObjOp) { return WalkResult::interrupt(); }).wasInterrupted();
        if (hasMatmul) {
            builder.create<emitc::IncludeOp>("lib/matmul_intf.h");
        }
        bool hasTensorDesc = mod.walk([](ascendc::TensorDescOp) { return WalkResult::interrupt(); }).wasInterrupted();
        if (hasTensorDesc) {
            builder.create<emitc::IncludeOp>("kernel_operator_list_tensor_intf.h");
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createGenerateBoilerplatePass()
{
    return std::make_unique<GenerateBoilerplatePass>();
}
