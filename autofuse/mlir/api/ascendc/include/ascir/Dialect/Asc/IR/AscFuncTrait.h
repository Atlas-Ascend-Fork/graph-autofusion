/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASC_IR_ASC_FUNC_TRAIT_H
#define ASCIR_DIALECT_ASC_IR_ASC_FUNC_TRAIT_H

namespace mlir {
namespace OpTrait {
template <typename ConcreteOp>
struct AscConstructorTrait : public TraitBase<ConcreteOp, AscConstructorTrait> {
    static mlir::LogicalResult verifyTrait(Operation* op)
    {
        if (op->getNumResults() != 1) {
            return op->emitError("AscConstructorTrait must have a result");
        }
        return success();
    }
};

template <typename ConcreteOp>
struct AscMemberFuncTrait : public TraitBase<ConcreteOp, AscMemberFuncTrait> {
    static mlir::LogicalResult verifyTrait(Operation* op)
    {
        if (op->getNumOperands() < 1 || op->getNumResults() > 1) {
            return op->emitError("AscMemberFuncTrait must have more than one inputs and less than one return value");
        }
        return success();
    }
};

template <typename ConcreteOp>
struct AscFuncTrait : public TraitBase<ConcreteOp, AscFuncTrait> {
    static mlir::LogicalResult verifyTrait(Operation* op)
    {
        if (op->getNumResults() > 1) {
            return op->emitError("AscFunc trait only support less than one return value");
        }
        return success();
    }
};

} // namespace OpTrait
} // namespace mlir

#endif