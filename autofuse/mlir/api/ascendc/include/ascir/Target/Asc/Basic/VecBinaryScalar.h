/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_BASIC_BINARY_SCALAR_INSTR_H
#define ASCIR_TARGET_ASC_BASIC_BINARY_SCALAR_INSTR_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Vector-scalar operations
//===----------------------------------------------------------------------===//

template <typename VecScalarL0Op>
auto printOperation(CodeEmitter& emitter, VecScalarL0Op op) -> LogicalResultForT<
    VecScalarL0Op, ascendc::AddsL0Op, ascendc::SubsL0Op, ascendc::LeakyReluL0Op, ascendc::MaxsL0Op, ascendc::MinsL0Op,
    ascendc::MulsL0Op, ascendc::DivsL0Op, ascendc::ShiftLeftL0Op, ascendc::ShiftRightL0Op>
{
    auto& os = emitter.ostream();
    FAIL_OR(printIsSetMaskTemplate(emitter, op));
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getScalar()) << ", " << emitter.getOrCreateName(op.getMask()) << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}

template <typename VecScalarL1Op>
auto printOperation(CodeEmitter& emitter, VecScalarL1Op op) -> LogicalResultForT<
    VecScalarL1Op, ascendc::AddsL1Op, ascendc::SubsL1Op, ascendc::LeakyReluL1Op, ascendc::MaxsL1Op, ascendc::MinsL1Op,
    ascendc::MulsL1Op, ascendc::DivsL1Op, ascendc::ShiftLeftL1Op, ascendc::ShiftRightL1Op>
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);
    FAIL_OR(printIsSetMaskTemplate(emitter, op));
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getScalar()) << ", " << maskName << ", "
       << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getRepeatParams()) << ")";
    return success();
}

template <typename VecScalarL2Op>
auto printOperation(CodeEmitter& emitter, VecScalarL2Op op) -> LogicalResultForT<
    VecScalarL2Op, ascendc::AddsL2Op, ascendc::SubsL2Op, ascendc::LeakyReluL2Op, ascendc::MaxsL2Op, ascendc::MinsL2Op,
    ascendc::MulsL2Op, ascendc::DivsL2Op, ascendc::ShiftLeftL2Op, ascendc::ShiftRightL2Op>
{
    auto& os = emitter.ostream();
    FAIL_OR(printIsSetMaskTemplate(emitter, op));
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getScalar()) << ", " << emitter.getOrCreateName(op.getCalCount()) << ")";
    return success();
}

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_BASIC_BINARY_SCALAR_INSTR_H
