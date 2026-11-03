/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_EMITASC_UTILS_INITSTRUCTBUILDER_H
#define ASCIR_DIALECT_EMITASC_UTILS_INITSTRUCTBUILDER_H

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/IR/Builders.h"

namespace mlir {
namespace emitasc {

class InitStructBuilder {
    SmallVector<StringRef> fieldNames;
    SmallVector<Value> fieldValues;

public:
    Type result;

    InitStructBuilder(Type result) : result(result) {}
    InitStructBuilder(const InitStructBuilder&) = default;
    InitStructBuilder(InitStructBuilder&&) = default;
    ~InitStructBuilder() = default;

    ArrayRef<StringRef> names() const { return fieldNames; }
    ValueRange values() const { return fieldValues; }

    InitStructBuilder& addField(StringRef name, Value value)
    {
        fieldNames.push_back(name);
        fieldValues.push_back(value);
        return *this;
    }

    InitStructOp create(OpBuilder& builder, Location loc) const
    {
        return builder.create<InitStructOp>(loc, result, builder.getStrArrayAttr(names()), values());
    }
};

} // namespace emitasc
} // namespace mlir

#endif // ASCIR_DIALECT_EMITASC_UTILS_INITSTRUCTBUILDER_H
