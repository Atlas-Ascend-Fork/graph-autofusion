/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_UTILS_CONSTANTOPBUILDER_H
#define ASCIR_DIALECT_UTILS_CONSTANTOPBUILDER_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace ascir {

struct ConstantOpBuilder {
    using ConstantOp = ::mlir::arith::ConstantOp;

    ConstantOpBuilder(OpBuilder& builder) : builder(builder) {}
    ~ConstantOpBuilder() = default;

    ConstantOp create(TypedAttr attr)
    {
        auto op = ConstantOp::create(builder, builder.getUnknownLoc(), attr);
        return op;
    }

    ConstantOp create(IndexType type, int64_t value) { return create(builder.getIndexAttr(value)); }

    ConstantOp create(IntegerType type, int64_t value) { return create(builder.getIntegerAttr(type, value)); }

    ConstantOp create(FloatType type, double value) { return create(builder.getFloatAttr(type, value)); }

    Value index(int64_t value) { return create(builder.getIndexType(), value); }

    Value i64(int64_t value) { return create(builder.getI64Type(), value); }

    Value i32(int64_t value) { return create(builder.getI32Type(), value); }

    Value i16(int64_t value) { return create(builder.getI16Type(), value); }

    Value i8(int64_t value) { return create(builder.getI8Type(), value); }

    Value i1(bool value) { return create(builder.getI1Type(), value); }

    Value f64(double value) { return create(builder.getF64Type(), value); }

    Value f32(float value) { return create(builder.getF32Type(), value); }

    Value f16(float value) { return create(builder.getF16Type(), value); }

    template <typename IntType>
    Value integer(IntType value)
    {
        return create(builder.getIntegerType(sizeof(IntType) * CHAR_BIT), value);
    }

private:
    OpBuilder& builder;
};

} // namespace ascir
} // namespace mlir

#endif // ASCIR_DIALECT_UTILS_CONSTANTOPBUILDER_H
