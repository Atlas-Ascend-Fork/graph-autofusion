/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_BASIC_REG_H
#define ASCIR_TARGET_ASC_BASIC_REG_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Binary register API operations
//===----------------------------------------------------------------------===//

template <typename BinaryRegOp>
LogicalResultForT<
    BinaryRegOp, ascendc::AddRegOp, ascendc::AndRegOp, ascendc::DivRegOp, ascendc::FusedAbsSubRegOp,
    ascendc::FusedExpSubRegOp, ascendc::FusedMulDstAddRegOp, ascendc::SubRegOp, ascendc::MaxRegOp, ascendc::MinRegOp,
    ascendc::MulRegOp, ascendc::MulAddDstRegOp, ascendc::OrRegOp, ascendc::PreluRegOp, ascendc::XorRegOp>
printOperation(CodeEmitter& emitter, BinaryRegOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDstReg()) << ", "
       << emitter.getOrCreateName(op.getSrc0Reg()) << ", " << emitter.getOrCreateName(op.getSrc1Reg()) << ", "
       << emitter.getOrCreateName(op.getMaskReg()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// Unary, Reduction, Duplicate register API operations
//===----------------------------------------------------------------------===//

template <typename RegOp>
LogicalResultForT<
    RegOp, ascendc::AbsRegOp, ascendc::ExpRegOp, ascendc::LnRegOp, ascendc::LogRegOp, ascendc::Log10RegOp,
    ascendc::MaskNotRegOp, ascendc::NegRegOp, ascendc::NotRegOp, ascendc::ReluRegOp, ascendc::SqrtRegOp,
    ascendc::ReduceMaxRegOp, ascendc::ReduceMinRegOp, ascendc::ReduceSumRegOp, ascendc::DuplicateRegOp>
printOperation(CodeEmitter& emitter, RegOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDstReg()) << ", "
       << emitter.getOrCreateName(op.getSrcReg()) << ", " << emitter.getOrCreateName(op.getMaskReg()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// VecScalar register API operations
//===----------------------------------------------------------------------===//

template <typename VecScalarOp>
LogicalResultForT<
    VecScalarOp, ascendc::AddsRegOp, ascendc::MulsRegOp, ascendc::MaxsRegOp, ascendc::MinsRegOp,
    ascendc::LeakyReluRegOp, ascendc::ShiftLeftsRegOp, ascendc::ShiftRightsRegOp>
printOperation(CodeEmitter& emitter, VecScalarOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDstReg()) << ", "
       << emitter.getOrCreateName(op.getSrcReg()) << ", " << emitter.getOrCreateName(op.getScalar()) << ", "
       << emitter.getOrCreateName(op.getMaskReg()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// Other register API operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::DataCopyLoadOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::DataCopyStoreOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::UpdateMaskOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::RegTensorOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::DuplicateScalarRegOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::LocalMemBarOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::SelectRegOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CreateMaskOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_BASIC_REG_H
