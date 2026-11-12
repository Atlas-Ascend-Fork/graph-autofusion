/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_OPS_H
#define ASCIR_TARGET_ASC_AUTOFUSE_OPS_H

#include "ascir/Target/Asc/Autofuse/AutofuseActivation.h"
#include "ascir/Target/Asc/Autofuse/AutofuseArith.h"
#include "ascir/Target/Asc/Autofuse/ControlFlow.h"
#include "ascir/Target/Asc/Autofuse/AutofuseDataCopy.h"
#include "ascir/Target/Asc/Autofuse/RegDataCopy.h"
#include "ascir/Target/Asc/Autofuse/Index.h"
#include "ascir/Target/Asc/Autofuse/AutofusePad.h"
#include "ascir/Target/Asc/Autofuse/AutofuseTranspose.h"
#include "ascir/Target/Asc/Autofuse/AutofuseVecBinary.h"
#include "ascir/Target/Asc/Autofuse/VecBroadcast.h"
#include "ascir/Target/Asc/Autofuse/AutofuseVecCmpsel.h"
#include "ascir/Target/Asc/Autofuse/VecRemovePad.h"
#include "ascir/Target/Asc/Autofuse/VecReduceND.h"
#include "ascir/Target/Asc/Autofuse/VecTernary.h"
#include "ascir/Target/Asc/Autofuse/AutofuseVecUnary.h"
#include "ascir/Target/Asc/Autofuse/AutofuseVecVconv.h"

#include <tuple>

namespace mlir {
namespace autofuse {

// Every operation whose printer this subtree provides. `Translation.cpp` runs a
// second `addCases` over this tuple -- the one that runs over its own tuple is
// unchanged, and neither list has to know about the other.
//
// Operations carrying `AscFunc` are not listed here: their printers are generated
// into `AscendCOpEmit.h.inc`, which upstream's tuple already splices in, and that
// generator reads `Autofuse/Ops.td` along with the rest of the dialect.
using PrintableOpTypes = std::tuple<
    // ControlFlow ops
    cf::AssertOp,
    // Arithmetic ops
    arith::AddUIExtendedOp, arith::MulSIExtendedOp,
    // Index ops
    index::AddOp, index::MulOp, index::MaxUOp, index::MinUOp, index::RemUOp, index::CeilDivUOp, index::DivUOp,
    index::CmpOp,
    // Activation operations
    ascendc::GeluOp, ascendc::SigmoidOp, ascendc::SigmoidExtendOp,
    // Data copy operations
    ascendc::DataCopyPadExtendOp, ascendc::DataCopyPadExtendRegBaseOp, ascendc::DataCopyNddmaRegBaseOp,
    // UB <-> 向量寄存器的对齐搬运（见 Autofuse/OpRegDataCopy.td）
    ascendc::LoadAlignRegOp, ascendc::StoreAlignRegOp,
    // BinaryL2Op on a stock API -- `HypotL2Op` is the one that also takes a tmp buffer
    ascendc::BitwiseAndL2Op, ascendc::HypotL2Op,
    // UnaryL2Op on a stock API -- the floating-point predicate family, plus `Log10`
    ascendc::IsNanL2Op, ascendc::IsFiniteL2Op, ascendc::IsInfL2Op, ascendc::Log10L2Op,
    // The one ternary stock API -- a tmp buffer too, see Autofuse/VecTernary.h
    ascendc::FmaL2Op,
    // API extension wrappers
    ascendc::AbsExtendOp, ascendc::NegExtendOp, ascendc::NegExtendRegBaseOp, ascendc::ReciprocalExtendOp,
    ascendc::IsnanExtendOp, ascendc::IsFiniteExtendOp, ascendc::IsInfExtendOp, ascendc::WhereContiguousOp,
    ascendc::WhereContiguousRegBaseOp, ascendc::ClipByValueContiguousOp, ascendc::ClipByValueContiguousRegBaseOp,
    ascendc::BitwiseAndExtendOp, ascendc::FloorDivExtendOp, ascendc::FloorDivExtendRegBaseOp,
    ascendc::RemainderExtendOp, ascendc::RemainderExtendRegBaseOp, ascendc::PowOp, ascendc::PowRegBaseOp,
    ascendc::SubExtendRegBaseOp,
    ascendc::DivExtendRegBaseOp, ascendc::LogicalAndExtendOp, ascendc::LogicalOrExtendOp,
    ascendc::LogicalAndExtendRegBaseOp, ascendc::LogicalOrExtendRegBaseOp,
    ascendc::LogicalAndScalarExtendOp, ascendc::LogicalOrScalarExtendOp,
    ascendc::LogicalAndExtendsRegBaseOp, ascendc::SignExtendOp, ascendc::SignExtendRegBaseOp,
    ascendc::AxpyExtendContiguousOp, ascendc::TrueDivExtendOp, ascendc::TanhExtendRegBaseOp,
    ascendc::ErfExtendRegBaseOp, ascendc::NdtrExtendRegBaseOp, ascendc::NdtriExtendRegBaseOp,
    ascendc::LogNdtrExtendRegBaseOp, ascendc::I0ExtendRegBaseOp, ascendc::AiryAiExtendRegBaseOp,
    ascendc::BesselJ0ExtendRegBaseOp, ascendc::BesselJ1ExtendRegBaseOp,
    ascendc::I0eExtendRegBaseOp, ascendc::I1eExtendRegBaseOp,
    ascendc::ModifiedBesselI0ExtendRegBaseOp, ascendc::ModifiedBesselI1ExtendRegBaseOp,
    ascendc::ModifiedBesselK0ExtendRegBaseOp, ascendc::ModifiedBesselK1ExtendRegBaseOp,
    ascendc::ErfinvExtendRegBaseOp, ascendc::Exp2ExtendRegBaseOp,
    ascendc::AcosExtendRegBaseOp, ascendc::AcoshExtendRegBaseOp,
    ascendc::AsinExtendRegBaseOp, ascendc::AsinhExtendRegBaseOp,
    ascendc::AtanExtendRegBaseOp, ascendc::AtanhExtendRegBaseOp,
    ascendc::CeilExtendRegBaseOp, ascendc::CosExtendRegBaseOp, ascendc::CoshExtendRegBaseOp,
    ascendc::ErfcxExtendRegBaseOp, ascendc::RoundExtendRegBaseOp,
    ascendc::SinExtendRegBaseOp, ascendc::SinhExtendRegBaseOp, ascendc::TanExtendRegBaseOp,
    ascendc::BesselY0ExtendRegBaseOp, ascendc::BesselY1ExtendRegBaseOp,
    ascendc::SphericalBesselJ0ExtendRegBaseOp,
    ascendc::ScaledModifiedBesselK0ExtendRegBaseOp, ascendc::ScaledModifiedBesselK1ExtendRegBaseOp,
    ascendc::Expm1ExtendRegBaseOp, ascendc::Log1pExtendRegBaseOp,
    // 多项式族：带一个 int64 模板实参（见 Autofuse/OpVecUnary.td）
    ascendc::ChebyshevPolynomialTExtendRegBaseOp, ascendc::ChebyshevPolynomialUExtendRegBaseOp,
    ascendc::ChebyshevPolynomialVExtendRegBaseOp, ascendc::ChebyshevPolynomialWExtendRegBaseOp,
    ascendc::ShiftedChebyshevPolynomialTExtendRegBaseOp, ascendc::ShiftedChebyshevPolynomialUExtendRegBaseOp,
    ascendc::ShiftedChebyshevPolynomialVExtendRegBaseOp, ascendc::ShiftedChebyshevPolynomialWExtendRegBaseOp,
    ascendc::CopySignExtendRegBaseOp, ascendc::TruncDivExtendRegBaseOp,
    ascendc::BitwiseOrExtendRegBaseOp, ascendc::BitwiseXorExtendRegBaseOp,
    ascendc::Atan2ExtendRegBaseOp, ascendc::ZetaExtendRegBaseOp,
    ascendc::IgammaExtendRegBaseOp, ascendc::IgammacExtendRegBaseOp,
    ascendc::LogicalXorExtendRegBaseOp, ascendc::FmodExtendRegBaseOp, ascendc::NextAfterExtendRegBaseOp,
    ascendc::RsqrtExtendOp, ascendc::LogicalNotExtendOp,
    ascendc::LogicalNotExtendRegBaseOp,
    ascendc::BitwiseNotExtendRegBaseOp,
    ascendc::SubsOp, ascendc::DivsOp, ascendc::TrueDivExtendsOp, ascendc::DivExtendsRegBaseOp,
    ascendc::CompareExtendContiguousOp, ascendc::CompareExtendContiguousRegBaseOp,
    ascendc::CompareScalarExtendContiguousOp, ascendc::CompareScalarExtendContiguousRegBaseOp,
    // 内联广播（brc_inline，custom emitter -- see VecBinary.cpp）
    ascendc::BinaryBrcInlineOp,
    // Type conversion operations
    ascendc::CastExtendContiguousOp, ascendc::CastExtendContiguousRegBaseOp, ascendc::CastRegOp,
    ascendc::Ceil2IntExtendRegBaseOp,
    // 谓词寄存器：VF 里的一条 `Reg::Compare` 与它写进去的那个 `MaskReg` 的声明
    // （`MaskRegOp` 是上游 Basic 的 op，上游的 printer 表里没有它，见 VecCmpsel.h）。
    ascendc::CompareRegOp, ascendc::MaskRegOp,
    // Vector broadcast operations (custom emitter -- see VecBroadcast.cpp)
    ascendc::BroadcastL2Op, ascendc::DuplicateU8Op, ascendc::BroadcastDim3Op, ascendc::BroadcastExtendRegBaseOp,
    // Vector layout operations (custom emitters -- see VecRemovePad.cpp / Transpose.cpp / Pad.cpp)
    ascendc::RemovePadExtendOp, ascendc::TransposeExtendRegBaseOp, ascendc::ConfusionTransposeOp,
    ascendc::PadOp, ascendc::DataCopyBlkAlignedOp,
    // Reduce family (custom emitters -- see VecReduceND.cpp)
    ascendc::ReduceSum2DL2Op, ascendc::ReduceSumInt32Op, ascendc::ReduceMax2DL2Op, ascendc::ReduceMin2DL2Op,
    ascendc::ReduceProd2DL2Op,
    ascendc::ReduceAny2DL2Op, ascendc::ReduceAll2DL2Op, ascendc::ReduceInitOp>;

// What the translation has to write into the module preamble on Autofuse's
// account, called once before the module's operations are emitted.
//
// API extension headers are not among it: operations declare them through
// `ApiExtensionOpInterface` and the kernel-source layer materializes one
// `emitc.include` per distinct header before translation runs.
void printModulePrelude(CodeEmitter& emitter, ModuleOp moduleOp);

} // namespace autofuse
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_OPS_H
