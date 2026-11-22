/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/EmitNameStack.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

using llvm::formatv;

EmitNameStack::EmitNameStack() { labelInScopeCount.push(0); }

void EmitNameStack::pushScope()
{
    for (auto& [_, scope] : valueInScopeCount) {
        scope.push(scope.top());
    }
    labelInScopeCount.push(labelInScopeCount.top());
}

void EmitNameStack::popScope()
{
    for (auto& [_, scope] : valueInScopeCount) {
        scope.pop();
    }
    labelInScopeCount.pop();
}

EmitNameStack::CountStack& EmitNameStack::getCountStack(const std::string& prefix)
{
    auto it = valueInScopeCount.find(prefix);
    if (it == valueInScopeCount.end()) {
        auto& stack = valueInScopeCount[prefix];
        for (size_t i = 0; i < labelInScopeCount.size(); i++)
            stack.push(0);
        return stack;
    }
    return it->second;
}

std::string EmitNameStack::getNameForEmission(Value val)
{
    auto getName = [&](const std::string& prefix) -> std::string {
        return prefix + std::to_string(++getCountStack(prefix).top());
    };
    auto getDefaultName = [&]() -> std::string { return getName("v"); };
    auto replaceUnsupportedChars = [](std::string str) {
        std::replace(str.begin(), str.end(), '.', '_');
        std::replace(str.begin(), str.end(), '-', 'm');
        std::replace(str.begin(), str.end(), '+', 'p');
        return str;
    };
    if (!val)
        return getName("NULL");
    if (auto defConstOp = val.getDefiningOp<arith::ConstantOp>()) {
        auto attr = defConstOp.getValue();
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr)) {
            if (defConstOp.getType().isIndex()) {
                return replaceUnsupportedChars(formatv("c{0}_idx", intAttr.getInt()));
            }
            return replaceUnsupportedChars(
                formatv("c{0}_i{1}", intAttr.getInt(), defConstOp.getType().getIntOrFloatBitWidth()));
        }
        if (auto fpAttr = dyn_cast_or_null<FloatAttr>(attr)) {
            SmallVector<char> number;
            fpAttr.getValue().toString(number);
            std::string typeAbbrev;
            llvm::raw_string_ostream os(typeAbbrev);
            defConstOp.getType().print(os);
            return replaceUnsupportedChars(formatv("c{0}_{1}", number, typeAbbrev));
        }
    }
    return getDefaultName();
}
