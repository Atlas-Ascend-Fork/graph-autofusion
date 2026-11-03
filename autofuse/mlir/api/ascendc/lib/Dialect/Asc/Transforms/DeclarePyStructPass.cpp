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
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"

#include <unordered_set>

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_DECLAREPYSTRUCT
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

using PyStructVector = SmallVector<emitasc::PyStructType>;

struct PyStructTypeHash {
    std::hash<const void*> h;
    size_t operator()(emitasc::PyStructType type) const { return h(type.getAsOpaquePointer()); }
};

PyStructVector deduplicate(ArrayRef<emitasc::PyStructType> pyStructs)
{
    PyStructVector result;
    std::unordered_set<emitasc::PyStructType, PyStructTypeHash> unique(pyStructs.begin(), pyStructs.end());
    for (auto pyStruct : pyStructs) {
        auto it = unique.find(pyStruct);
        if (it == unique.end())
            continue;
        result.push_back(pyStruct);
        unique.erase(it);
    }
    return result;
}

void collectPyStructTypes(PyStructVector& structs, Value& arg)
{
    arg.getType().walk([&](Type type) {
        if (auto pyStructType = dyn_cast<emitasc::PyStructType>(type)) {
            structs.push_back(pyStructType);
        }
    });
}

class DeclarePyStructPass : public ascendc::impl::DeclarePyStructBase<DeclarePyStructPass> {
public:
    void runOnOperation() override
    {
        auto mod = getOperation();
        PyStructVector structs;
        mod.walk([&](Operation* op) {
            for (auto& region : op->getRegions()) {
                for (auto& block : region) {
                    for (auto& arg : block.getArguments()) {
                        collectPyStructTypes(structs, arg);
                    }
                }
            }

            for (auto result : op->getResults()) {
                collectPyStructTypes(structs, result);
            }
        });

        auto pyStructs = deduplicate(structs);
        auto builder = ImplicitLocOpBuilder::atBlockBegin(mod->getLoc(), mod.getBody());
        for (auto pyStruct : pyStructs) {
            builder.create<emitasc::DeclarePyStructOp>(pyStruct);
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createDeclarePyStructPass() { return std::make_unique<DeclarePyStructPass>(); }
