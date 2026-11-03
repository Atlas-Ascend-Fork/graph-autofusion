/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_VEC_UNARY_H
#define ASCIR_TARGET_ASC_AUTOFUSE_VEC_UNARY_H

#include "ascir/Target/Asc/Basic/VecUnary.h"

namespace mlir {
namespace ascendc {

// 浮点谓词族的标准调用：`AscendC::IsNan` 这一批写的是 `LocalTensor<bool>`，而 AFIR
// 那一层的结果是 ui8，所以目的地要就地 ReinterpretCast 一下。legacy 先声明一个
// `LocalTensor<bool> y_cast` 再下标，这里是先下标再 cast——ui8 和 bool 都是一个字节，
// 元素偏移一样，效果相同，少一条语句。
template <typename PredicateL2Op>
auto printOperation(CodeEmitter& emitter, PredicateL2Op op)
    -> LogicalResultForT<PredicateL2Op, ascendc::IsNanL2Op, ascendc::IsFiniteL2Op, ascendc::IsInfL2Op>
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst())
       << ".template ReinterpretCast<bool>(), " << emitter.getOrCreateName(op.getSrc()) << ", "
       << "static_cast<uint32_t>(" << emitter.getOrCreateName(op.getCalCount()) << "))";
    return success();
}

// `Log10L2Op` is an Autofuse operation on a stock AscendC API, so its emitter is
// the stock L2 unary one -- whose type list lives in an upstream header this
// subtree does not edit. A second overload carrying a one-element list does the
// job: upstream's template SFINAEs out for this operation, so only this one is
// viable and the call is not ambiguous. Same shape as `BitwiseAndL2Op` in
// VecBinary.h.
template <typename UnaryL2Op>
auto printOperation(CodeEmitter& emitter, UnaryL2Op op) -> LogicalResultForT<UnaryL2Op, ascendc::Log10L2Op>
{
    emitter.ostream() << ascNamespace << "::" << op.getAPIName();
    printUnaryL2Params(emitter, op);
    return success();
}

//===----------------------------------------------------------------------===//
// Autofuse API extension wrappers
//===----------------------------------------------------------------------===//

// Shared body for every Autofuse `XxxExtend` wrapper. Two things differ from the
// stock unary emitter above and hold for all of them: the call is unqualified,
// because the wrappers are global functions rather than members of `AscendC::`,
// and the element type is spelled out as an explicit template argument rather
// than deduced, matching how legacy codegen emits these. Pass a null
// `sharedTmpBuffer` for the wrappers that take no tmp buffer, and a null
// `constBlock` for the ones that take no pre-filled constant block.
LogicalResult printApiExtensionUnary(CodeEmitter& emitter, Operation* op, StringRef apiName, Value dst, Value src,
                                     Value sharedTmpBuffer, Value calCount, Value constBlock = Value());

// The wrappers that take a caller-provided tmp buffer: `Xxx(dst, src, tmp, count)`.
template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::AbsExtendOp, ascendc::ReciprocalExtendOp,
                         ascendc::IsnanExtendOp, ascendc::IsFiniteExtendOp, ascendc::IsInfExtendOp,
                         ascendc::SignExtendOp, ascendc::SignExtendRegBaseOp, ascendc::TanhExtendRegBaseOp,
                         ascendc::ErfExtendRegBaseOp, ascendc::NdtrExtendRegBaseOp,
                         ascendc::NdtriExtendRegBaseOp, ascendc::LogNdtrExtendRegBaseOp,
                         ascendc::I0ExtendRegBaseOp, ascendc::AiryAiExtendRegBaseOp,
                         ascendc::BesselJ0ExtendRegBaseOp, ascendc::BesselJ1ExtendRegBaseOp,
                         ascendc::I0eExtendRegBaseOp, ascendc::I1eExtendRegBaseOp,
                         ascendc::ModifiedBesselI0ExtendRegBaseOp, ascendc::ModifiedBesselI1ExtendRegBaseOp,
                         ascendc::ModifiedBesselK0ExtendRegBaseOp, ascendc::ModifiedBesselK1ExtendRegBaseOp,
                         ascendc::Exp2ExtendRegBaseOp,
                         ascendc::AcosExtendRegBaseOp, ascendc::AcoshExtendRegBaseOp,
                         ascendc::AsinExtendRegBaseOp, ascendc::AsinhExtendRegBaseOp,
                         ascendc::AtanExtendRegBaseOp, ascendc::AtanhExtendRegBaseOp,
                         ascendc::CeilExtendRegBaseOp, ascendc::CosExtendRegBaseOp,
                         ascendc::CoshExtendRegBaseOp, ascendc::ErfcxExtendRegBaseOp,
                         ascendc::RoundExtendRegBaseOp, ascendc::SinExtendRegBaseOp,
                         ascendc::SinhExtendRegBaseOp, ascendc::TanExtendRegBaseOp,
                         ascendc::BesselY0ExtendRegBaseOp, ascendc::BesselY1ExtendRegBaseOp,
                         ascendc::SphericalBesselJ0ExtendRegBaseOp,
                         ascendc::ScaledModifiedBesselK0ExtendRegBaseOp,
                         ascendc::ScaledModifiedBesselK1ExtendRegBaseOp>
{
    return printApiExtensionUnary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getSharedTmpBuffer(),
                                  op.getCalCount());
}

// The wrappers that take none: `Xxx(dst, src, count)`.
template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::NegExtendOp, ascendc::NegExtendRegBaseOp,
                         ascendc::LogicalNotExtendRegBaseOp,
                         ascendc::ErfinvExtendRegBaseOp,
                         ascendc::Expm1ExtendRegBaseOp, ascendc::Log1pExtendRegBaseOp,
                         ascendc::BitwiseNotExtendRegBaseOp>
{
    return printApiExtensionUnary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), /*sharedTmpBuffer=*/Value(),
                                  op.getCalCount());
}

// The polynomial-family wrappers: `Xxx<T, N>(dst, src, tmp, count)`. The extra
// int64 template argument sits *after* the element type, so it cannot reuse
// `printApiExtensionUnary` -- a template argument list cannot be written out of
// order. Same shape as `isScalarLatter` on the scalar binary wrappers.
LogicalResult printApiExtensionUnaryTemplateAttr(CodeEmitter& emitter, Operation* op, StringRef apiName, Value dst,
                                                 Value src, Value sharedTmpBuffer, Value calCount,
                                                 int64_t templateArgument);

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::ChebyshevPolynomialTExtendRegBaseOp,
                         ascendc::ChebyshevPolynomialUExtendRegBaseOp,
                         ascendc::ChebyshevPolynomialVExtendRegBaseOp,
                         ascendc::ChebyshevPolynomialWExtendRegBaseOp,
                         ascendc::ShiftedChebyshevPolynomialTExtendRegBaseOp,
                         ascendc::ShiftedChebyshevPolynomialUExtendRegBaseOp,
                         ascendc::ShiftedChebyshevPolynomialVExtendRegBaseOp,
                         ascendc::ShiftedChebyshevPolynomialWExtendRegBaseOp>
{
    return printApiExtensionUnaryTemplateAttr(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(),
                                              op.getSharedTmpBuffer(), op.getCalCount(), op.getN());
}

// The wrappers that take both a pre-filled constant block and a tmp buffer:
// `Xxx(dst, src, blk, tmp, count)`.
template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::RsqrtExtendOp, ascendc::LogicalNotExtendOp>
{
    return printApiExtensionUnary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getSharedTmpBuffer(),
                                  op.getCalCount(), op.getConstBlock());
}

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_VEC_UNARY_H
