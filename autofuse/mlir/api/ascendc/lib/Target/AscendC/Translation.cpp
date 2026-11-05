/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Translation.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/Adv/AdvActivation.h"
#include "ascir/Target/Asc/Adv/AdvBroadcast.h"
#include "ascir/Target/Asc/Adv/Kfc.h"
#include "ascir/Target/Asc/Adv/Math.h"
#include "ascir/Target/Asc/Adv/Matmul.h"
#include "ascir/Target/Asc/Adv/Normalization.h"
#include "ascir/Target/Asc/Adv/Quantization.h"
#include "ascir/Target/Asc/Adv/Reduction.h"
#include "ascir/Target/Asc/Basic/Aipp.h"
#include "ascir/Target/Asc/Basic/BlockSync.h"
#include "ascir/Target/Asc/Basic/DataConversion.h"
#include "ascir/Target/Asc/Basic/BasicDataCopy.h"
#include "ascir/Target/Asc/Basic/DumpTensor.h"
#include "ascir/Target/Asc/Basic/ListTensor.h"
#include "ascir/Target/Asc/Basic/OtherOps.h"
#include "ascir/Target/Asc/Basic/Reg.h"
#include "ascir/Target/Asc/Basic/Scalar.h"
#include "ascir/Target/Asc/Basic/SwapMem.h"
#include "ascir/Target/Asc/Basic/VecBilinearInterpolation.h"
#include "ascir/Target/Asc/Basic/BasicVecBinary.h"
#include "ascir/Target/Asc/Basic/VecBinaryScalar.h"
#include "ascir/Target/Asc/Basic/BasicVecCmpsel.h"
#include "ascir/Target/Asc/Basic/VecDuplicate.h"
#include "ascir/Target/Asc/Basic/VecGather.h"
#include "ascir/Target/Asc/Basic/VecGatherMask.h"
#include "ascir/Target/Asc/Basic/VecReduce.h"
#include "ascir/Target/Asc/Basic/VecScatter.h"
#include "ascir/Target/Asc/Basic/VecTernaryScalar.h"
#include "ascir/Target/Asc/Basic/BasicVecUnary.h"
#include "ascir/Target/Asc/Basic/BasicVecVconv.h"
#include "ascir/Target/Asc/Core/GlobalTensor.h"
#include "ascir/Target/Asc/Core/LocalTensor.h"
#include "ascir/Target/Asc/Core/ShapeInfo.h"
#include "ascir/Target/Asc/EmitAscTarget.h"
#include "ascir/Target/Asc/External/ExternalArith.h"
#include "ascir/Target/Asc/External/Emitc.h"
#include "ascir/Target/Asc/External/Func.h"
#include "ascir/Target/Asc/External/ExternalMath.h"
#include "ascir/Target/Asc/External/MemRef.h"
#include "ascir/Target/Asc/External/Scf.h"
#include "ascir/Target/Asc/Fwk/TBuf.h"
#include "ascir/Target/Asc/Fwk/TQue.h"
#include "ascir/Target/Asc/UniversalEmitter.h"

// Autofuse subtree seam: the operations it adds printers for, and what it puts
// in the module preamble. Everything else it contributes reaches this file
// through the generated `AscendCOpEmit.h.inc` below.
#include "ascir/Target/Asc/Autofuse/Ops.h"

using namespace mlir;

namespace {
LogicalResult printOperation(CodeEmitter& emitter, ModuleOp moduleOp)
{
    CodeEmitter::Scope scope(emitter);
    autofuse::printModulePrelude(emitter, moduleOp);

    for (Operation& op : moduleOp) {
        if (failed(emitOperation(emitter, op, /*trailingSemicolon=*/false))) {
            return failure();
        }
    }
    return success();
}

LogicalResult printOperation(CodeEmitter&, ascendc::NoOp) { return success(); }

using PrintableOpTypes = std::tuple<
    // Common register API operations
    ascendc::CreateMaskOp, ascendc::ReduceMaxRegOp, ascendc::ReduceMinRegOp, ascendc::ReduceSumRegOp,
    ascendc::DuplicateRegOp, ascendc::DataCopyStoreOp, ascendc::DataCopyLoadOp, ascendc::UpdateMaskOp,
    ascendc::RegTensorOp, ascendc::LocalMemBarOp, ascendc::DuplicateScalarRegOp, ascendc::SelectRegOp,
    // Binary register API operations
    ascendc::AddRegOp, ascendc::AndRegOp, ascendc::DivRegOp, ascendc::FusedAbsSubRegOp, ascendc::FusedExpSubRegOp,
    ascendc::FusedMulDstAddRegOp, ascendc::SubRegOp, ascendc::MaxRegOp, ascendc::MinRegOp, ascendc::MulRegOp,
    ascendc::MulAddDstRegOp, ascendc::OrRegOp, ascendc::PreluRegOp, ascendc::XorRegOp,
    // Unary register API operations
    ascendc::AbsRegOp, ascendc::ExpRegOp, ascendc::LnRegOp, ascendc::LogRegOp, ascendc::Log10RegOp,
    ascendc::MaskNotRegOp, ascendc::NegRegOp, ascendc::NotRegOp, ascendc::ReluRegOp, ascendc::SqrtRegOp,
    // VecScalar register API operations
    ascendc::AddsRegOp, ascendc::MulsRegOp, ascendc::MaxsRegOp, ascendc::MinsRegOp, ascendc::LeakyReluRegOp,
    ascendc::ShiftLeftsRegOp, ascendc::ShiftRightsRegOp,
    // Builtin ops
    ModuleOp,
    // EmitC ops
    emitc::CastOp, emitc::ConstantOp, emitc::IncludeOp, emitc::VariableOp, emitc::VerbatimOp,
    // Func ops
    func::CallOp, func::ConstantOp, func::FuncOp, func::ReturnOp,
    // SCF ops
    scf::ConditionOp, scf::ForOp, scf::IfOp, scf::IndexSwitchOp, scf::WhileOp, scf::YieldOp,
    // MemRef ops
    memref::AllocaOp, memref::CastOp, memref::LoadOp, memref::StoreOp,
    // Arithmetic ops
    arith::ConstantOp, arith::MulIOp, arith::DivSIOp, arith::RemSIOp, arith::AddIOp, arith::IndexCastOp, arith::CmpIOp,
    arith::CmpFOp, arith::SubIOp, arith::CeilDivSIOp, arith::SelectOp, arith::AndIOp, arith::ShLIOp, arith::ShRSIOp,
    arith::MaximumFOp, arith::MaxNumFOp, arith::AddFOp, arith::MinimumFOp, arith::MinNumFOp, arith::DivFOp,
    arith::MinSIOp, arith::MaxSIOp, arith::MulFOp, arith::SubFOp, arith::ExtUIOp, arith::ExtSIOp, arith::ExtFOp,
    arith::TruncIOp, arith::TruncFOp, arith::FPToSIOp, arith::FPToUIOp, arith::SIToFPOp, arith::UIToFPOp,
    arith::BitcastOp, arith::ShRUIOp, arith::OrIOp, arith::XOrIOp, arith::DivUIOp, arith::MulUIExtendedOp,
    arith::NegFOp,
    // Math ops
    math::AbsFOp, math::CopySignOp, math::SqrtOp, math::Atan2Op, math::ExpOp, math::LogOp, math::Log2Op, math::ErfOp,
    math::CosOp, math::SinOp, math::CeilOp, math::FloorOp, math::RsqrtOp, math::Exp2Op, math::FmaOp, math::RoundOp,
    // EmitAsc operations
    emitasc::CallOpaqueOp, emitasc::DeclarePyStructOp, emitasc::DereferenceOp, emitasc::MaskOp, emitasc::MemberOp,
    emitasc::MemberPtrOp, emitasc::MemberRefOp, emitasc::PtrOffsetOp, emitasc::ReinterpretCastOp, emitasc::SetMemberOp,
    emitasc::VariableOp, emitasc::VerbatimOp, emitasc::CopyStructOp, emitasc::InitStructOp, emitasc::VecScopeOp,
    emitasc::VFForOp, emitasc::YieldOp,
    // Adv
    // Activation operations
    ascendc::SimpleSoftMaxOp, ascendc::SoftMaxOp, ascendc::SoftmaxFlashV2Op, ascendc::SwiGLUOp,
    // Kfc operations
    ascendc::KfcInitOp, ascendc::KfcInitObjOp, ascendc::KfcIsRunOp, ascendc::KfcRunOp, ascendc::KfcQuitOp,
    // Math
    // UnaryMathOp
    ascendc::AcoshOp, ascendc::AcosOp, ascendc::AsinhOp, ascendc::AsinOp, ascendc::AtanhOp, ascendc::AtanOp,
    ascendc::CeilOp, ascendc::CoshOp, ascendc::CosOp, ascendc::DigammaOp, ascendc::ErfcOp, ascendc::ErfOp,
    ascendc::ExpOp, ascendc::FloorOp, ascendc::FracOp, ascendc::LgammaOp, ascendc::LogOp, ascendc::Log2Op,
    ascendc::RoundOp, ascendc::SignOp, ascendc::SinhOp, ascendc::SinOp, ascendc::TanhOp, ascendc::TanOp,
    ascendc::TruncOp,
    // BinaryMathOp
    ascendc::PowerOp, ascendc::XorOp,
    // Other math
    ascendc::AxpyOp, ascendc::ClampMaxOp, ascendc::ClampMinOp, ascendc::CumSumOp,
    // Matmul operations
    ascendc::MatmulInitOp, ascendc::MatmulEndOp, ascendc::MatmulGetMatmulApiTilingOp, ascendc::RegistMatmulObjOp,
    // Normalization
    ascendc::RmsNormOp, ascendc::LayerNormOp,
    // Quantization
    ascendc::QuantOp,
    // Broadcast
    ascendc::BroadcastOp,
    // Reduction
    ascendc::ReduceOp,
    // Basic
    // AIPP operations
    ascendc::SetAippFunctionsOp,
    // Block synchronization operations
    ascendc::PipeBarrierOp, ascendc::WaitFlagOp, ascendc::CrossCoreSetFlagOp, ascendc::CrossCoreWaitFlagOp,
    ascendc::GetBufOp, ascendc::RlsBufOp,
    // DataConversion operations
    ascendc::TransposeOp, ascendc::TransposeExtOp, ascendc::TransDataTo5HDTensorListOp,
    ascendc::TransDataTo5HDUintListOp, ascendc::TransDataTo5HDOp, ascendc::TransDataTo5HDTensorOp,
    // Data copy operations
    ascendc::CopyL0Op, ascendc::CopyL1Op, ascendc::DataCopyL0Op, ascendc::DataCopyL2Op, ascendc::DataCopyNd2NzOp,
    ascendc::DataCopyNz2NdOp, ascendc::DataCopySliceOp, ascendc::DataCopyCO12DstOp, ascendc::NdDmaParamsOp,
    // Dump tensor operations
    ascendc::PrintfOp,
    // TensorDesc operations
    ascendc::TensorDescOp, ascendc::TensorDescSetShapeAddrOp,
    // ListTensorDesc operations
    ascendc::ListTensorDescV2Op, ascendc::ListTensorDescGetDataPtrOp,
    // Other operations
    ascendc::ConstructOp, ascendc::AscendIsAICOp, ascendc::AscendIsAIVOp, ascendc::IfAICOp, ascendc::IfAIVOp,
    ascendc::YieldOp, LLVM::UndefOp, ascendc::FftsCrossCoreSyncOp, ascendc::SetFftsBaseAddrOp,
    ascendc::PopStackBufferOp, ascendc::MrgSortOp, ascendc::SortOp, ascendc::FixpipeOp, ascendc::FixpipeWithWorkspaceOp,
    ascendc::GetMrgSortResultOp, ascendc::GetStoreAtomicConfigOp,
    // Scalar operations
    ascendc::ScalarCastOp,
    // Swap and workspace operations
    ascendc::GetSysWorkspacePtrOp, ascendc::SetSysWorkspaceOp,
    // Vector bilinear interpolation
    ascendc::BilinearInterpolationL0Op, ascendc::BilinearInterpolationL1Op,
    // Vector binary operations
    // BinaryL0Op
    ascendc::AddL0Op, ascendc::AddDeqReluL0Op, ascendc::AddReluL0Op, ascendc::AddReluCastL0Op, ascendc::AndL0Op,
    ascendc::DivL0Op, ascendc::FusedMulAddL0Op, ascendc::FusedMulAddReluL0Op, ascendc::MaxL0Op, ascendc::MinL0Op,
    ascendc::MulL0Op, ascendc::MulAddDstL0Op, ascendc::MulCastL0Op, ascendc::OrL0Op, ascendc::SubL0Op,
    ascendc::SubReluL0Op, ascendc::SubReluCastL0Op,
    // BinaryL1Op
    ascendc::AddL1Op, ascendc::AddDeqReluL1Op, ascendc::AddReluL1Op, ascendc::AddReluCastL1Op, ascendc::AndL1Op,
    ascendc::DivL1Op, ascendc::FusedMulAddL1Op, ascendc::FusedMulAddReluL1Op, ascendc::MaxL1Op, ascendc::MinL1Op,
    ascendc::MulL1Op, ascendc::MulAddDstL1Op, ascendc::MulCastL1Op, ascendc::OrL1Op, ascendc::SubL1Op,
    ascendc::SubReluL1Op, ascendc::SubReluCastL1Op,
    // BinaryL2Op
    ascendc::AddL2Op, ascendc::AddDeqReluL2Op, ascendc::AddReluL2Op, ascendc::AddReluCastL2Op, ascendc::AndL2Op,
    ascendc::DivL2Op, ascendc::FusedAbsSubL2Op, ascendc::FusedExpSubL2Op, ascendc::FusedMulAddL2Op,
    ascendc::FusedMulAddReluL2Op, ascendc::MaxL2Op, ascendc::MinL2Op, ascendc::MulL2Op, ascendc::MulAddDstL2Op,
    ascendc::MulCastL2Op, ascendc::OrL2Op, ascendc::PreluL2Op, ascendc::SubL2Op, ascendc::SubReluL2Op,
    ascendc::SubReluCastL2Op,
    // BinaryL3Op
    ascendc::AddL3Op, ascendc::DivL3Op, ascendc::MulL3Op, ascendc::SubL3Op,
    // Block Reduce operations
    ascendc::BlockReduceSumL1Op, ascendc::BlockReduceMaxL1Op, ascendc::BlockReduceMinL1Op,
    // Vector binary scalar operations
    // other vector binary operators
    ascendc::GatherbL0Op, ascendc::GatherL0Op, ascendc::GatherL1Op, ascendc::GatherL2Op,
    ascendc::BilinearInterpolationL0Op, ascendc::BilinearInterpolationL1Op,
    // VecScalarL0Op
    ascendc::AddsL0Op, ascendc::SubsL0Op, ascendc::LeakyReluL0Op, ascendc::MaxsL0Op, ascendc::MinsL0Op,
    ascendc::MulsL0Op, ascendc::DivsL0Op, ascendc::ShiftLeftL0Op, ascendc::ShiftRightL0Op,
    // VecScalarL1Op
    ascendc::AddsL1Op, ascendc::SubsL1Op, ascendc::LeakyReluL1Op, ascendc::MaxsL1Op, ascendc::MinsL1Op,
    ascendc::MulsL1Op, ascendc::DivsL1Op, ascendc::ShiftLeftL1Op, ascendc::ShiftRightL1Op,
    // VecScalarL2Op
    ascendc::AddsL2Op, ascendc::SubsL2Op, ascendc::LeakyReluL2Op, ascendc::MaxsL2Op, ascendc::MinsL2Op,
    ascendc::MulsL2Op, ascendc::DivsL2Op, ascendc::ShiftLeftL2Op, ascendc::ShiftRightL2Op,
    // VectorTernaryScalarL0Op
    ascendc::AxpyL0Op,
    // VectorTernaryScalarL1Op
    ascendc::AxpyL1Op,
    // VectorTernaryScalarL2Op
    ascendc::AxpyL2Op,
    // VecCmpSel (Select) operations
    ascendc::CompareL1Op, ascendc::CompareRL1Op, ascendc::CompareScalarL1Op, ascendc::SelectScalarL1Op,
    ascendc::SelectL1Op,
    // Duplicate operations
    ascendc::DuplicateL0Op, ascendc::DuplicateL1Op, ascendc::DuplicateL2Op,
    // Vector gather operations
    ascendc::GatherbL0Op, ascendc::GatherL0Op, ascendc::GatherL2Op,
    // Vector mask operations
    ascendc::SetVectorMaskL0Op, ascendc::SetVectorMaskL1Op,
    // Vector Reduce operations
    ascendc::PairReduceSumL1Op, ascendc::WholeReduceMaxL1Op, ascendc::WholeReduceMinL1Op, ascendc::WholeReduceSumL1Op,
    ascendc::ReduceMaxL1Op, ascendc::ReduceMinL1Op, ascendc::ReduceSumL1Op,
    // Vector scatter operations
    ascendc::ScatterL1Op,
    // Vector unary operations
    // UnaryL0Op
    ascendc::AbsL0Op, ascendc::ExpL0Op, ascendc::LnL0Op, ascendc::NotL0Op, ascendc::ReciprocalL0Op, ascendc::ReluL0Op,
    ascendc::RsqrtL0Op, ascendc::SqrtL0Op,
    // UnaryL1Op
    ascendc::AbsL1Op, ascendc::ExpL1Op, ascendc::LnL1Op, ascendc::NotL1Op, ascendc::ReciprocalL1Op, ascendc::ReluL1Op,
    ascendc::RsqrtL1Op, ascendc::SqrtL1Op,
    // UnaryL2Op
    ascendc::AbsL2Op, ascendc::ExpL2Op, ascendc::LnL2Op, ascendc::NotL2Op, ascendc::ReciprocalL2Op, ascendc::ReluL2Op,
    ascendc::RsqrtL2Op, ascendc::SqrtL2Op, ascendc::NegL2Op,
    // VecVcon (Type conversion) operations
    ascendc::CastL0Op, ascendc::CastL1Op, ascendc::CastL2Op, ascendc::CastDeqL0Op, ascendc::CastDeqL1Op,
    ascendc::CastDeqL2Op, ascendc::SetDeqScaleL4Op,
    // Vector gatherMask operations
    ascendc::GatherMaskOp, ascendc::GetGatherMaskRemainCountOp,

    // Core
    // GlobalTensor operations
    ascendc::GlobalTensorSubIndexOp, ascendc::GlobalTensorBracketOp,
    // LocalTensor operations
    ascendc::LocalTensorV2Op, ascendc::LocalTensorV3Op, ascendc::LocalTensorBracketOp,
    ascendc::LocalTensorReinterpretCastOp, ascendc::LocalTensorSubIndexOp, ascendc::LocalTensorGetPhyAddrV2Op,
    // ShapeInfo
    ascendc::ShapeInfoShapeOp, ascendc::ShapeInfoOriginalShapeOp,

    // Fwk
    // BaseQueue operations (TQue, TQueBind)
    ascendc::TQueBindAllocTensorOp, ascendc::TQueBindAllocTensorInPlaceOp, ascendc::TQueBindDequeTensorOp,
    ascendc::TQueBindDequeTensorPosOp, ascendc::TQueBindDequeTensorInPlaceOp, ascendc::TQueBindEnqueTensorPosOp,
    ascendc::ToQueBindOp,
    // Buffer operations (TBuf)
    ascendc::TBufGetTensorOp, ascendc::TBufGetWithOffsetOp,

#define GET_OP_TYPE_LIST
#include "ascir/Dialect/Asc/IR/AscendCOpEmit.h.inc"
    // Sentinel
    ascendc::NoOp>;

template <typename TypeSwitchT, typename OpT, typename CallbackT>
void addCaseByType(TypeSwitchT& typeSwitch, CallbackT&& callback)
{
    typeSwitch.template Case<OpT>([&callback](auto op) { return callback(op); });
}

template <typename TypeSwitchT, typename... OpTypes, typename CallbackT>
void addCasesImpl(TypeSwitchT& typeSwitch, std::tuple<OpTypes...>*, CallbackT&& callback)
{
    using addCaseFunc = void (*)(TypeSwitchT&, CallbackT&);
    static addCaseFunc caseFunc[] = {&addCaseByType<TypeSwitchT, OpTypes, CallbackT>...};
    for (size_t i = 0; i < sizeof...(OpTypes); ++i) {
        caseFunc[i](typeSwitch, callback);
    }
}

template <typename TupleT, typename TypeSwitchT, typename CallbackT>
void addCases(TypeSwitchT& typeSwitch, const CallbackT& callback)
{
    addCasesImpl(typeSwitch, static_cast<TupleT*>(nullptr), callback);
}
} // namespace

namespace mlir {
namespace ascendc {
#define GET_OP_PRINT_FUNC_LIST
#include "ascir/Dialect/Asc/IR/AscendCOpEmit.cpp.inc"
} // namespace ascendc
} // namespace mlir
LogicalResult emitOperation(CodeEmitter& emitter, Operation& op, bool trailingSemicolon)
{
    if (auto apiOp = dyn_cast<ascendc::APIOp>(op)) {
        auto comment = apiOp.getComment();
        if (!comment.empty()) {
            emitter.ostream() << "// " << comment << "\n";
        }
    }
    llvm::TypeSwitch<Operation*, LogicalResult> typeSwitch(&op);
    auto callback = [&](auto opNode) -> LogicalResult {
        return printOperation(emitter, opNode);
    };

    addCases<PrintableOpTypes>(typeSwitch, callback);
    addCases<autofuse::PrintableOpTypes>(typeSwitch, callback);
    LogicalResult status = typeSwitch.Default(
        [&](Operation* op) -> LogicalResult { return op->emitOpError("unable to find printer for op"); });
    if (failed(status)) {
        return failure();
    }
    emitter.ostream() << (trailingSemicolon ? ";\n" : "\n");
    return success();
}

LogicalResult mlir::translateToAscendC(Operation* op, raw_ostream& os)
{
    CodeEmitter emitter(os);
    return emitOperation(emitter, *op, /*trailingSemicolon=*/false);
}
