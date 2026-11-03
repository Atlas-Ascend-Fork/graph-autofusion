/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/OtherOps.h"

using namespace mlir;
using namespace mlir::ascendc;

namespace {

struct AippMemberInfo {
    const char* aippMemberName;
    const std::vector<const char*> subMemberNames;
};

const AippMemberInfo* getAippMemberInfo(size_t index)
{
    static const std::vector<AippMemberInfo> memberInfos = {
        {"paddingParams", {"paddingMode", "paddingValueCh0", "paddingValueCh1", "paddingValueCh2", "paddingValueCh3"}},
        {"swapParams", {"isSwapRB", "isSwapUV", "isSwapAX"}},
        {"singleLineParams", {"isSingleLineCopy"}},
        {"dtcParams",
         {"dtcMeanCh0", "dtcMeanCh1", "dtcMeanCh2", "dtcMinCh0", "dtcMinCh1", "dtcMinCh2", "dtcVarCh0", "dtcVarCh1",
          "dtcVarCh2", "dtcRoundMode"}},
        {"cPaddingParams", {"cPaddingMode", "cPaddingValue"}},
        {"cscParams",
         {"isEnableCsc", "cscMatrixR0C0", "cscMatrixR0C1", "cscMatrixR0C2", "cscMatrixR1C0", "cscMatrixR1C1",
          "cscMatrixR1C2", "cscMatrixR2C0", "cscMatrixR2C1", "cscMatrixR2C2", "cscBiasIn0", "cscBiasIn1", "cscBiasIn2",
          "cscBiasOut0", "cscBiasOut1", "cscBiasOut2"}}};

    if (index >= memberInfos.size()) {
        return nullptr;
    }

    return &memberInfos[index];
}

LogicalResult printAippMemberAssignment(CodeEmitter& emitter, ascendc::ConstructOp op, size_t memberIndex)
{
    auto& os = emitter.ostream();

    const AippMemberInfo* memberInfoPtr = getAippMemberInfo(memberIndex);

    if (!memberInfoPtr) {
        return op.emitError("Internal Error: Index out of bounds when accessing AippMemberInfo for member index ")
               << memberIndex;
    }

    auto subStructOp = dyn_cast_or_null<ascendc::ConstructOp>(op->getOperand(memberIndex).getDefiningOp());
    if (!subStructOp) {
        return op.emitError("Internal Error: Expected operand of AippParams to be a ConstructOp.");
    }

    if (subStructOp->getNumOperands() != memberInfoPtr->subMemberNames.size()) {
        return op.emitError("Internal Error: Mismatch in operand count for AIPP member ")
               << memberInfoPtr->aippMemberName;
    }

    for (size_t i = 0; i < subStructOp->getNumOperands(); ++i) {
        os << emitter.getOrCreateName(op->getResult(0)) << "." << memberInfoPtr->aippMemberName << "."
           << memberInfoPtr->subMemberNames[i] << " = " << emitter.getOrCreateName(subStructOp->getOperand(i)) << ";\n";
    }

    return success();
}

LogicalResult printAippStructConstruction(CodeEmitter& emitter, ascendc::ConstructOp op)
{
    auto& os = emitter.ostream();
    mlir::Type resultType = op->getResult(0).getType();

    return llvm::TypeSwitch<mlir::Type, LogicalResult>(resultType)
        .Case([&](ascendc::AippParamsType) -> LogicalResult {
            constexpr size_t paddingParamsOpIndex = 0;
            constexpr size_t templateTypeMemberIndex = 1;
            if (op->getNumOperands() <= paddingParamsOpIndex) {
                return op.emitError("Internal Error: AippParams is missing operands.");
            }
            auto paddingConstructOp =
                dyn_cast_or_null<ascendc::ConstructOp>(op->getOperand(paddingParamsOpIndex).getDefiningOp());
            if (!paddingConstructOp || paddingConstructOp->getNumOperands() <= templateTypeMemberIndex) {
                return op.emitError("Internal Error: Cannot deduce template type from paddingParams.");
            }
            mlir::Type templateType = paddingConstructOp->getOperand(templateTypeMemberIndex).getType();

            os << "AscendC::AippParams<";
            if (failed(emitter.emitType(op.getLoc(), templateType))) {
                return failure();
            }
            os << "> " << emitter.getOrCreateName(op->getResult(0)) << ";\n";

            for (size_t i = 0; i < op->getNumOperands(); ++i) {
                if (failed(printAippMemberAssignment(emitter, op, i))) {
                    return failure();
                }
            }

            return success();
        })
        .Case<
            ascendc::AippPaddingParamsType, ascendc::AippSwapParamsType, ascendc::AippSingleLineParamsType,
            ascendc::AippDataTypeConvParamsType, ascendc::AippChannelPaddingParamsType,
            ascendc::AippColorSpaceConvParamsType>([&](auto) -> LogicalResult { return success(); })
        .Default([](auto) -> LogicalResult { return failure(); });
}

} // namespace

//===----------------------------------------------------------------------===//
// Other operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ConstructOp op)
{
    if (succeeded(printAippStructConstruction(emitter, op))) {
        return success();
    }

    auto& os = emitter.ostream();
    if (op.getIsStatic()) {
        os << "static ";
    }
    if (op.getIsConstexpr()) {
        os << "constexpr ";
    }
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    if (op.getNumOperands() == 0) {
        return success();
    }
    os << '{';
    auto emitOperand = [&os, &emitter](Value operand, Type type) {
        if (operand.getType() == type) {
            os << emitter.getOrCreateName(operand);
            return;
        }
        if (isa<MemRefType>(type)) {
            os << "reinterpret_cast<";
        } else {
            os << "static_cast<";
        }
        (void)emitter.emitType(operand.getLoc(), type);
        os << ">(" << emitter.getOrCreateName(operand) << ")";
    };
    SmallVector<Type> types;
    if (auto typesAttr = op.getTypesAttr()) {
        if (typesAttr.size() != op->getNumOperands()) {
            return emitError(op.getLoc(), "Expect the size of typesAttr equals to numbers of operands");
        }
        llvm::transform(typesAttr, std::back_inserter(types), [](Attribute a) { return cast<TypeAttr>(a).getValue(); });
    } else {
        types.append(op->getOperandTypes().begin(), op->getOperandTypes().end());
    }
    llvm::interleaveComma(llvm::zip_equal(op.getOperands(), types), os, [&emitOperand](auto pair) {
        const auto& [operand, type] = pair;
        emitOperand(operand, type);
    });
    os << '}';
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::FftsCrossCoreSyncOp op)
{
    auto& os = emitter.ostream();
    os << "ffts_cross_core_sync(" << ascendc::stringifyEnum(op.getPipe()).upper() << ", "
       << emitter.getOrCreateName(op.getConfig()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GetMrgSortResultOp op)
{
    auto& os = emitter.ostream();
    Value mrgSortList1Value = op.getMrgSortList1();
    Value mrgSortList2Value = op.getMrgSortList2();
    Value mrgSortList3Value = op.getMrgSortList3();
    Value mrgSortList4Value = op.getMrgSortList4();

    os << "uint16_t " << emitter.getOrCreateName(mrgSortList1Value) << ";\n";
    os << "uint16_t " << emitter.getOrCreateName(mrgSortList2Value) << ";\n";
    os << "uint16_t " << emitter.getOrCreateName(mrgSortList3Value) << ";\n";
    os << "uint16_t " << emitter.getOrCreateName(mrgSortList4Value) << ";\n";

    os << ascNamespace << "::" << op.getAPIName();
    os << "(" << emitter.getOrCreateName(mrgSortList1Value);
    os << ", " << emitter.getOrCreateName(mrgSortList2Value);
    os << ", " << emitter.getOrCreateName(mrgSortList3Value);
    os << ", " << emitter.getOrCreateName(mrgSortList4Value) << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::MrgSortOp op)
{
    static int elementCountListCounter = 0;
    auto uniqueId = std::to_string(elementCountListCounter++);
    auto& os = emitter.ostream();
    auto elementCountListName = (emitter.getOrCreateName(op.getDst()) + "_element_count_list_" + uniqueId).str();
    auto sortedNumName = (emitter.getOrCreateName(op.getDst()) + "_sorted_num_" + uniqueId).str();
    os << "uint16_t " << elementCountListName << "[] = {";
    llvm::interleaveComma(op.getElementCountList(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";
    os << "uint32_t " << sortedNumName << "[] = {";
    llvm::interleaveComma(op.getSortedNum(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";
    os << ascNamespace << "::" << op.getAPIName();
    auto tensorType = cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    os << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), tensorType));
    os << ", " << op.getIsExhaustedSuspension() << ">"
       << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSortList()) << ", "
       << elementCountListName << ", " << sortedNumName << ", " << emitter.getOrCreateName(op.getValidBit()) << ", "
       << emitter.getOrCreateName(op.getRepeatTime()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SortOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName();
    auto tensorType = cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    os << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), tensorType));
    os << ", " << op.getIsFullSort() << ">"
       << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getConcat()) << ", "
       << emitter.getOrCreateName(op.getIndex()) << ", " << emitter.getOrCreateName(op.getTmp()) << ", "
       << emitter.getOrCreateName(op.getRepeatTime()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::PopStackBufferOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getTensor().getType().getElementType()));
    os << ", ";
    CodeEmitter::emitTPosition(os, op.getPos());
    os << ">(" << emitter.getOrCreateName(op.getTensor()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SetFftsBaseAddrOp op)
{
    auto& os = emitter.ostream();
    os << "set_ffts_base_addr(*" << emitter.getOrCreateName(op.getOperand()) << ")";
    return success();
}

LogicalResult mlir::printOperation(CodeEmitter& emitter, LLVM::UndefOp op)
{
    return emitter.emitVariableDeclaration(op->getResult(0), false);
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ResetMaskOp op);

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::FixpipeOp op)
{
    auto& os = emitter.ostream();
    FAIL_OR(printFixpipeTemplate(emitter, op));
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getIntriParams()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::FixpipeWithWorkspaceOp op)
{
    auto& os = emitter.ostream();
    FAIL_OR(printFixpipeTemplate(emitter, op));
    os << "(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getCbufWorkspace()) << ", " << emitter.getOrCreateName(op.getIntriParams()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GetStoreAtomicConfigOp op)
{
    auto& os = emitter.ostream();
    Value atomicTypeValue = op.getAtomicType();
    Value atomicOpValue = op.getAtomicOp();

    os << "uint16_t " << emitter.getOrCreateName(atomicTypeValue) << ";\n";
    os << "uint16_t " << emitter.getOrCreateName(atomicOpValue) << ";\n";

    os << ascNamespace << "::" << op.getAPIName();
    os << "(" << emitter.getOrCreateName(atomicTypeValue);
    os << ", " << emitter.getOrCreateName(atomicOpValue) << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::IfAICOp op)
{
    auto& os = emitter.ostream();
    for (OpResult result : op.getResults()) {
        if (failed(emitter.emitVariableDeclaration(result, /*trailingSemicolon=*/true))) {
            return failure();
        }
    }
    os << "if ASCEND_IS_AIC {\n";
    os.indent();
    for (auto& operation : op.getRegion().front()) {
        FAIL_OR(emitOperation(emitter, operation, needsSemicolon(operation)));
    }
    os.unindent() << "}";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::IfAIVOp op)
{
    auto& os = emitter.ostream();
    for (OpResult result : op.getResults()) {
        if (failed(emitter.emitVariableDeclaration(result, /*trailingSemicolon=*/true))) {
            return failure();
        }
    }
    os << "if ASCEND_IS_AIV {\n";
    os.indent();
    for (auto& operation : op.getRegion().front()) {
        FAIL_OR(emitOperation(emitter, operation, needsSemicolon(operation)));
    }
    os.unindent() << "}";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::YieldOp yieldOp)
{
    auto& os = emitter.ostream();
    Operation* parentOp = yieldOp->getParentOp();
    llvm::interleave(
        llvm::zip(parentOp->getResults(), yieldOp.getOperands()),
        [&](auto pair) {
            auto& [result, operand] = pair;
            os << emitter.getOrCreateName(result) << " = " << emitter.getOrCreateName(operand) << ";";
        },
        [&] { os << "\n"; });
    return success();
}
