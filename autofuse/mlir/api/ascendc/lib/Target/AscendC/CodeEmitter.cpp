/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/CodeEmitter.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/Common.h"
#include "ascir/Target/Asc/TargetUtils.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using llvm::formatv;

namespace {
constexpr uint32_t dtypeBitWidth1 = 1;
constexpr uint32_t dtypeBitWidth8 = 8;
constexpr uint32_t dtypeBitWidth16 = 16;
constexpr uint32_t dtypeBitWidth32 = 32;
constexpr uint32_t dtypeBitWidth64 = 64;
constexpr uint32_t smallStringLength = 128;
constexpr uint32_t batchmodeNoneValue = 0;
constexpr uint32_t batchmodeBatchLessThanL1Value = 1;
constexpr uint32_t batchmodeBatchLargeThanL1Value = 2;
constexpr uint32_t batchmodeSingleLargeThanL1Value = 3;
constexpr uint32_t iterateorderOrderMValue = 0;
constexpr uint32_t iterateorderOrderNValue = 1;
constexpr uint32_t iterateorderUndefValue = 2;
constexpr uint32_t scheduletypeInnerProductValue = 0;
constexpr uint32_t scheduletypeOuterProductValue = 1;
constexpr uint32_t batchoutmodeSingleBatchValue = 0;
constexpr uint32_t batchoutmodeMultiBatchOne = 1;
constexpr uint32_t batchoutmodeDynamicValue = 2;
} // namespace

void CodeEmitter::emitTPosition(raw_ostream& os, ascendc::TPosition pos)
{
    if (pos == ascendc::TPosition::GM)
        os << ascNamespace << "::TPosition::GM";
    else if (pos == ascendc::TPosition::A1)
        os << ascNamespace << "::TPosition::A1";
    else if (pos == ascendc::TPosition::A2)
        os << ascNamespace << "::TPosition::A2";
    else if (pos == ascendc::TPosition::B1)
        os << ascNamespace << "::TPosition::B1";
    else if (pos == ascendc::TPosition::B2)
        os << ascNamespace << "::TPosition::B2";
    else if (pos == ascendc::TPosition::C1)
        os << ascNamespace << "::TPosition::C1";
    else if (pos == ascendc::TPosition::C2)
        os << ascNamespace << "::TPosition::C2";
    else if (pos == ascendc::TPosition::CO1)
        os << ascNamespace << "::TPosition::CO1";
    else if (pos == ascendc::TPosition::CO2)
        os << ascNamespace << "::TPosition::CO2";
    else if (pos == ascendc::TPosition::VECIN)
        os << ascNamespace << "::TPosition::VECIN";
    else if (pos == ascendc::TPosition::VECOUT)
        os << ascNamespace << "::TPosition::VECOUT";
    else if (pos == ascendc::TPosition::VECCALC)
        os << ascNamespace << "::TPosition::VECCALC";
    else
        llvm_unreachable("unexpected ascendc::TPosition value");
}

void CodeEmitter::emitCO2Layout(raw_ostream& os, ascendc::CO2Layout layout)
{
    if (layout == ascendc::CO2Layout::NZ)
        os << ascNamespace << "::CO2Layout::NZ";
    else if (layout == ascendc::CO2Layout::ROW_MAJOR)
        os << ascNamespace << "::CO2Layout::ROW_MAJOR";
    else if (layout == ascendc::CO2Layout::COLUMN_MAJOR)
        os << ascNamespace << "::CO2Layout::COLUMN_MAJOR";
    else
        llvm_unreachable("unexpected ascendc::CO2Layout value");
}

void CodeEmitter::emitCubeFormat(raw_ostream& os, ascendc::CubeFormat format)
{
    if (format == ascendc::CubeFormat::ND)
        os << "CubeFormat::ND";
    else if (format == ascendc::CubeFormat::NZ)
        os << "CubeFormat::NZ";
    else if (format == ascendc::CubeFormat::ZN)
        os << "CubeFormat::ZN";
    else if (format == ascendc::CubeFormat::ZZ)
        os << "CubeFormat::ZZ";
    else if (format == ascendc::CubeFormat::NN)
        os << "CubeFormat::NN";
    else if (format == ascendc::CubeFormat::ND_ALIGN)
        os << "CubeFormat::ND_ALIGN";
    else if (format == ascendc::CubeFormat::SCALAR)
        os << "CubeFormat::SCALAR";
    else if (format == ascendc::CubeFormat::VECTOR)
        os << "CubeFormat::VECTOR";
    else
        llvm_unreachable("unexpected ascendc::CubeFormat value");
}

void CodeEmitter::emitLayoutMode(raw_ostream& os, ascendc::LayoutMode layout)
{
    if (layout == ascendc::LayoutMode::NONE)
        os << "LayoutMode::NONE";
    else if (layout == ascendc::LayoutMode::NORMAL)
        os << "LayoutMode::NORMAL";
    else if (layout == ascendc::LayoutMode::BSNGD)
        os << "LayoutMode::BSNGD";
    else if (layout == ascendc::LayoutMode::SBNGD)
        os << "LayoutMode::SBNGD";
    else if (layout == ascendc::LayoutMode::BNGS1S2)
        os << "LayoutMode::BNGS1S2";
    else
        llvm_unreachable("unexpected ascendc::LayoutMode value");
}

CodeEmitter::CodeEmitter(raw_ostream& os) : os(os)
{
    createTypeEmitMapper();
    createAttributeEmitMapper();
}

void CodeEmitter::createTypeEmitMapper()
{
    emitTypeMapper[TypeID::get<IndexType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitIndexType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<TensorType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitTensorType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<emitc::PointerType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitEmitcPointerType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<emitc::OpaqueType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitEmitcOpaqueType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::TBufType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscTBufType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::TBufPoolType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscTBufPoolType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::QueueType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscQueueType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::QueBindType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscQueBindType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::FixpipeParamsType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscFixpipeParamsType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::FixpipeParamsC310Type>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscFixpipeParamsC310Type(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::GlobalTensorType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscGlobalTensorType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::MatmulType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscMatmulType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::LocalMemAllocatorType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscLocalMemAllocatorType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::LocalTensorType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscLocalTensorType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::RegTensorType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscRegTensorType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::MaskRegType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscMaskRegType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<emitasc::PyStructType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscPyStructType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::DataCopyPadExtParamsType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscDataCopyPadExtParamsType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::MrgSortSrcListType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscMrgSortSrcListType(loc, type, flag);
    };
    emitTypeMapper[TypeID::get<ascendc::NdDmaParamsType>()] = [this](Location loc, Type type, bool flag) {
        return this->emitAscNdDmaParams(loc, type, flag);
    };
}

void CodeEmitter::createAttributeEmitMapper()
{
    emitAttributeMapper[TypeID::get<FloatAttr>()] = [this](Location loc, Attribute attr) {
        return this->emitFloatAttr(loc, attr);
    };
    emitAttributeMapper[TypeID::get<IntegerAttr>()] = [this](Location loc, Attribute attr) {
        return this->emitIntegerAttr(loc, attr);
    };
    emitAttributeMapper[TypeID::get<emitc::OpaqueAttr>()] = [this](Location loc, Attribute attr) {
        return this->emitEmitcOpaqueAttr(loc, attr);
    };
    emitAttributeMapper[TypeID::get<SymbolRefAttr>()] = [this](Location loc, Attribute attr) {
        return this->emitSymbolRefAttr(loc, attr);
    };
    emitAttributeMapper[TypeID::get<TypeAttr>()] = [this](Location loc, Attribute attr) {
        return this->emitTypeAttr(loc, attr);
    };
}

LogicalResult CodeEmitter::emitFloatAttr(Location /*loc*/, Attribute attr)
{
    auto fAttr = dyn_cast<FloatAttr>(attr);
    printFloat(fAttr.getValue());
    return success();
}

LogicalResult CodeEmitter::emitIntegerAttr(Location /*loc*/, Attribute attr)
{
    auto iAttr = dyn_cast<IntegerAttr>(attr);
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
        printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
        return success();
    }
    if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
        printInt(iAttr.getValue(), false);
        return success();
    }
    return success();
}

LogicalResult CodeEmitter::emitEmitcOpaqueAttr(Location /*loc*/, Attribute attr)
{
    auto oAttr = dyn_cast<emitc::OpaqueAttr>(attr);
    os << oAttr.getValue();
    return success();
}

LogicalResult CodeEmitter::emitSymbolRefAttr(Location loc, Attribute attr)
{
    auto sAttr = dyn_cast<SymbolRefAttr>(attr);
    if (sAttr.getNestedReferences().size() > 1)
        return emitError(loc, "attribute has more than 1 nested reference");
    os << sAttr.getRootReference().getValue();
    return success();
}

LogicalResult CodeEmitter::emitTypeAttr(Location loc, Attribute attr)
{
    auto type = dyn_cast<TypeAttr>(attr);
    return emitType(loc, type.getValue());
}

/// Return the existing or a new name for a Value.
StringRef CodeEmitter::getOrCreateName(Value val)
{
    if (!valueMapper.count(val)) {
        std::string name = nameStack.getNameForEmission(val);
        if (auto loc = val.getLoc()->findInstanceOf<NameLoc>())
            name = formatv("{0}_{1}", name, loc.getName().getValue()).str();
        valueMapper.insert(val, name);
    }
    return *valueMapper.begin(val);
}

/// Return the existing or a new label for a Block.
StringRef CodeEmitter::getOrCreateName(Block& block)
{
    if (!blockMapper.count(&block))
        blockMapper.insert(&block, formatv("label{0}", ++nameStack.labelInScopeCount.top()));
    return *blockMapper.begin(&block);
}

bool CodeEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val)
{
    switch (val) {
        case IntegerType::Signless:
        case IntegerType::Signed:
            return false;
        case IntegerType::Unsigned:
            return true;
    }
    llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

bool CodeEmitter::hasValueInScope(Value val) { return valueMapper.count(val); }

bool CodeEmitter::hasBlockLabel(Block& block) { return blockMapper.count(&block); }

void CodeEmitter::printInt(const APInt& value, bool isUnsigned)
{
    if (value.getBitWidth() == 1) {
        if (value.getBoolValue())
            os << "true";
        else
            os << "false";
    } else {
        constexpr uint32_t toStringLen = 10;
        SmallString<smallStringLength> strValue;
        value.toString(strValue, toStringLen, !isUnsigned, false);
        os << strValue;
    }
}

void CodeEmitter::printFloat(const APFloat& value)
{
    if (value.isFinite()) {
        SmallString<smallStringLength> strValue;
        // Use default values of toString except don't truncate zeros.
        value.toString(strValue, 0, 0, false);
        switch (llvm::APFloatBase::SemanticsToEnum(value.getSemantics())) {
            case llvm::APFloatBase::S_IEEEsingle:
                os << "(float)";
                break;
            case llvm::APFloatBase::S_BFloat:
                // bisheng compiler doesn't support explicit cast to bf16
                break;
            case llvm::APFloatBase::S_IEEEdouble:
                os << "(double)";
                break;
            default:
                break;
        };
        os << strValue;
    } else if (value.isNaN()) {
        os << "(0.f / 0.f) /* nan */";
    } else if (value.isInfinity()) {
        if (value.isNegative())
            os << "-";
        os << "__builtin_inff()";
    }
}

LogicalResult CodeEmitter::emitAttribute(Location loc, Attribute attr)
{
    auto it = emitAttributeMapper.find(attr.getTypeID());
    if (it != emitAttributeMapper.end()) {
        return it->second(loc, attr);
    }
    return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult CodeEmitter::emitOperands(Operation& op)
{
    auto emitOperand = [&](Value result) -> LogicalResult {
        if (!hasValueInScope(result))
            return op.emitOpError() << "operand value not in scope";
        os << getOrCreateName(result);
        return success();
    };
    return interleaveCommaWithError(op.getOperands(), os, emitOperand);
}

LogicalResult CodeEmitter::emitVariableDeclaration(OpResult opResult, bool trailingSemicolon)
{
    if (hasValueInScope(opResult)) {
        return opResult.getDefiningOp()->emitError("result variable for the operation already declared");
    }
    if (failed(emitType(opResult.getOwner()->getLoc(), opResult.getType())))
        return failure();
    os << " " << getOrCreateName(opResult);
    if (trailingSemicolon)
        os << ";\n";
    return success();
}

LogicalResult CodeEmitter::emitAssignPrefix(Operation& op)
{
    switch (op.getNumResults()) {
        case 0:
            break;
        case 1: {
            OpResult result = op.getResult(0);
            if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
                return failure();
            os << " = ";
            break;
        }
        default:
            llvm_unreachable("emission for multiple results is not implemented");
    }
    return success();
}

LogicalResult CodeEmitter::emitLabel(Block& block)
{
    if (!hasBlockLabel(block))
        return block.getParentOp()->emitError("label for block not found");
    // Add feature in `raw_indented_ostream` to ignore indent for block
    // label instead of using `getOStream`.
    os.getOStream() << getOrCreateName(block) << ":\n";
    return success();
}

void CodeEmitter::emitAddressSpace(ascendc::AddressSpace addressSpace)
{
    switch (addressSpace) {
        case ascendc::AddressSpace::Default:
            // print nothing
            break;
        case ascendc::AddressSpace::gm:
            os << "__gm__ ";
            break;
        case ascendc::AddressSpace::ca:
            os << "__ca__ ";
            break;
        case ascendc::AddressSpace::cb:
            os << "__cb__ ";
            break;
        case ascendc::AddressSpace::cc:
            os << "__cc__ ";
            break;
        case ascendc::AddressSpace::ubuf:
            os << "__ubuf__ ";
            break;
        case ascendc::AddressSpace::cbuf:
            os << "__cbuf__ ";
            break;
        case ascendc::AddressSpace::fbuf:
            os << "__fbuf__ ";
            break;
    }
}

LogicalResult CodeEmitter::emitIndexType(Location /*loc*/, Type /*type*/, bool /*emitAsUnsigned*/)
{
    return (os << "uint32_t"), success();
}

LogicalResult CodeEmitter::emitTensorType(Location loc, Type type, bool /*emitAsUnsigned*/)
{
    auto tensorType = dyn_cast<TensorType>(type);
    if (!tensorType.hasRank())
        return emitError(loc, "cannot emit unranked tensor type");
    if (!tensorType.hasStaticShape())
        return emitError(loc, "cannot emit tensor type with non static shape");
    os << "Tensor<";
    if (failed(emitType(loc, tensorType.getElementType())))
        return failure();
    auto shape = tensorType.getShape();
    for (auto dimSize : shape) {
        os << ", ";
        os << dimSize;
    }
    os << ">";
    return success();
}

LogicalResult CodeEmitter::emitEmitcPointerType(Location loc, Type type, bool /*emitAsUnsigned*/)
{
    auto pType = dyn_cast<emitc::PointerType>(type);
    if (failed(emitType(loc, pType.getPointee())))
        return failure();
    os << "*";
    return success();
}

LogicalResult CodeEmitter::emitEmitcOpaqueType(Location /*loc*/, Type type, bool /*emitAsUnsigned*/)
{
    auto oType = dyn_cast<emitc::OpaqueType>(type);
    os << oType.getValue();
    return success();
}

LogicalResult CodeEmitter::emitAscTBufType(Location /*loc*/, Type type, bool /*emitAsUnsigned*/)
{
    auto bType = dyn_cast<ascendc::TBufType>(type);
    os << ascNamespace << "::TBuf<";
    emitTPosition(os, bType.getTPosition());
    os << ">";
    return success();
}

LogicalResult CodeEmitter::emitAscTBufPoolType(Location /*loc*/, Type type, bool /*emitAsUnsigned*/)
{
    auto bType = dyn_cast<ascendc::TBufPoolType>(type);
    os << ascNamespace << "::TBufPool<";
    emitTPosition(os, bType.getTPosition());
    os << ", " << bType.getBufIDSize() << ">";
    return success();
}

LogicalResult CodeEmitter::emitAscQueueType(Location /*loc*/, Type type, bool /*emitAsUnsigned*/)
{
    auto qType = dyn_cast<ascendc::QueueType>(type);
    os << ascNamespace << "::TQue<";
    emitTPosition(os, qType.getPosition());
    os << ", " << qType.getDepth() << ">";
    return success();
}

LogicalResult CodeEmitter::emitAscQueBindType(Location /*loc*/, Type type, bool /*emitAsUnsigned*/)
{
    auto qType = dyn_cast<ascendc::QueBindType>(type);
    os << ascNamespace << "::TQueBind<";
    emitTPosition(os, qType.getSrcPosition());
    os << ", ";
    emitTPosition(os, qType.getDstPosition());
    os << ", " << qType.getDepth() << ">";
    return success();
}

LogicalResult CodeEmitter::emitAscFixpipeParamsType(Location loc, Type type, bool /*emitAsUnsigned*/)
{
    auto fpType = dyn_cast<ascendc::FixpipeParamsType>(type);
    auto instanceType = fpType.getType();
    os << ascNamespace << "::FixpipeParams<";
    if (failed(emitType(loc, instanceType)))
        return failure();
    os << '>';
    return success();
}

LogicalResult CodeEmitter::emitAscFixpipeParamsC310Type(Location /*loc*/, Type type, bool /*emitAsUnsigned*/)
{
    auto fpType = dyn_cast<ascendc::FixpipeParamsC310Type>(type);
    os << ascNamespace << "::FixpipeParamsC310<";
    emitCO2Layout(os, fpType.getCO2Layout());
    os << '>';
    return success();
}

LogicalResult CodeEmitter::emitAscGlobalTensorType(Location loc, Type type, bool /*emitAsUnsigned*/)
{
    auto pType = dyn_cast<ascendc::GlobalTensorType>(type);
    auto elemTy = pType.getElementType();
    os << ascNamespace << "::GlobalTensor<";
    if (failed(emitType(loc, elemTy)))
        return failure();
    os << ">";
    return success();
}

LogicalResult CodeEmitter::emitAscLocalMemAllocatorType(Location /*loc*/, Type type, bool /*emitAsUnsigned*/)
{
    auto pType = dyn_cast<ascendc::LocalMemAllocatorType>(type);
    os << ascNamespace << "::LocalMemAllocator<";
    uint64_t hardware = pType.getHardware();
    if (hardware == 0)
        os << ascNamespace << "::Hardware::GM";
    else if (hardware == 1)
        os << ascNamespace << "::Hardware::UB";
    else if (hardware == 2)
        os << ascNamespace << "::Hardware::L1";
    else if (hardware == 3)
        os << ascNamespace << "::Hardware::L0A";
    else if (hardware == 4)
        os << ascNamespace << "::Hardware::L0B";
    else if (hardware == 5)
        os << ascNamespace << "::Hardware::L0C";
    else if (hardware == 6)
        os << ascNamespace << "::Hardware::BIAS";
    else if (hardware == 7)
        os << ascNamespace << "::Hardware::FIXBUF";
    else if (hardware == 8)
        os << ascNamespace << "::Hardware::MAX";
    else
        llvm_unreachable("unexpected LocalMemAllocator hardware value");
    os << ">";
    return success();
}

LogicalResult CodeEmitter::emitAscLocalTensorType(Location loc, Type type, bool /*emitAsUnsigned*/)
{
    auto pType = dyn_cast<ascendc::LocalTensorType>(type);
    auto elemTy = pType.getElementType();
    os << ascNamespace << "::LocalTensor<";
    if (failed(emitType(loc, elemTy)))
        return failure();
    os << ">";
    return success();
}

LogicalResult CodeEmitter::emitAscRegTensorType(Location loc, Type type, bool /*emitAsUnsigned*/)
{
    auto rType = dyn_cast<ascendc::RegTensorType>(type);
    auto elemTy = rType.getElementType();
    os << ascNamespace << "::Reg::RegTensor<";
    if (failed(emitType(loc, elemTy)))
        return failure();
    os << ">";
    return success();
}

LogicalResult CodeEmitter::emitAscMaskRegType(Location /*loc*/, Type /*type*/, bool /*emitAsUnsigned*/)
{
    os << ascNamespace << "::Reg::MaskReg";
    return success();
}

void CodeEmitter::emitMatmulConfig(raw_ostream& os, ascendc::MatmulConfigAttr config, StringRef configName)
{
    os << "constexpr static MatmulConfig " << configName << "{";
    os << config.getDoNorm().getValue();
    os << ",";
    os << config.getDoBasicBlock().getValue();
    os << ",";
    os << config.getDoMultiDataLoad().getValue();
    os << ",";
    os << config.getBasicM().getValue();
    os << ",";
    os << config.getBasicN().getValue();
    os << ",";
    os << config.getBasicK().getValue();
    os << ",";
    os << config.getIntrinsicsCheck().getValue();
    os << ",";
    os << config.getIsNBatch().getValue();
    os << ",";
    os << config.getEnVecNd2nz().getValue();
    os << ",";
    os << config.getDoSpecialBasicBlock().getValue();
    os << ",";
    os << config.getDoMte2Preload().getValue();
    os << ",";
    os << config.getSingleCoreM().getValue();
    os << ",";
    os << config.getSingleCoreN().getValue();
    os << ",";
    os << config.getSingleCoreK().getValue();
    os << ",";
    os << config.getStepM().getValue();
    os << ",";
    os << config.getStepN().getValue();
    os << ",";
    os << config.getBaseMn().getValue();
    os << ",";
    os << config.getSingleCoreMn().getValue();
    os << ",";
    os << config.getEnUnitFlag().getValue();
    os << ",";
    os << config.getIsPerTensor().getValue();
    os << ",";
    os << config.getHasAntiQuantOffset().getValue();
    os << ",";
    os << config.getDoIbShareNorm().getValue();
    os << ",";
    os << config.getDoSpecialMdl().getValue();
    os << ",";
    os << config.getEnableInit().getValue();
    os << ",";
    if (config.getBatchMode().getValue() == batchmodeNoneValue)
        os << "BatchMode::NONE,";
    else if (config.getBatchMode().getValue() == batchmodeBatchLessThanL1Value)
        os << "BatchMode::BATCH_LESS_THAN_L1,";
    else if (config.getBatchMode().getValue() == batchmodeBatchLargeThanL1Value)
        os << "BatchMode::BATCH_LARGE_THAN_L1,";
    else if (config.getBatchMode().getValue() == batchmodeSingleLargeThanL1Value)
        os << "BatchMode::SINGLE_LARGE_THAN_L1,";
    else
        os << ",";
    os << config.getEnableEnd().getValue();
    os << ",";
    os << config.getEnableGetTensorC().getValue();
    os << ",";
    os << config.getEnableSetOrgShape().getValue();
    os << ",";
    os << config.getEnableSetBias().getValue();
    os << ",";
    os << config.getEnableSetTail().getValue();
    os << ",";
    os << config.getEnableQuantVector().getValue();
    os << ",";
    os << config.getEnableSetDefineData().getValue();
    os << ",";
    os << config.getIterateMode().getValue();
    os << ",";
    os << config.getEnableReuse().getValue();
    os << ",";
    os << config.getEnableUbReuse().getValue();
    os << ",";
    os << config.getEnableL1CacheUb().getValue();
    os << ",";
    os << config.getIntraBlockPartSum().getValue();
    os << ",";
    if (config.getIterateOrder().getValue() == iterateorderOrderMValue)
        os << "IterateOrder::ORDER_M,";
    else if (config.getIterateOrder().getValue() == iterateorderOrderNValue)
        os << "IterateOrder::ORDER_N,";
    else if (config.getIterateOrder().getValue() == iterateorderUndefValue)
        os << "IterateOrder::UNDEF,";
    else
        os << ",";
    if (config.getScheduleType().getValue() == scheduletypeInnerProductValue)
        os << "ScheduleType::INNER_PRODUCT,";
    else if (config.getScheduleType().getValue() == scheduletypeOuterProductValue)
        os << "ScheduleType::OUTER_PRODUCT,";
    else
        os << ",";
    os << config.getEnableDoubleCache().getValue();
    os << ",";
    os << config.getIsBiasBatch().getValue();
    os << ",";
    os << config.getEnableStaticPadZeros().getValue();
    os << ",";
    os << config.getIsPartialOutput().getValue();
    os << ",";
    os << config.getEnableMixDualMaster().getValue();
    os << ",";
    os << config.getIsA2b2Shared().getValue();
    os << ",";
    os << config.getIsEnableChannelSplit().getValue();
    os << ",";
    os << config.getEnableKdimReorderLoad().getValue();
    os << ",";
    os << config.getIsCo1Shared().getValue();
    os << ",";
    os << config.getSharedCo1BufferSize().getValue();
    os << ",";
    if (config.getBatchOutMode().getValue() == batchoutmodeSingleBatchValue)
        os << "BatchOutMode::SINGLE_BATCH";
    else if (config.getBatchOutMode().getValue() == batchoutmodeMultiBatchOne)
        os << "BatchOutMode::MULTI_BATCH";
    else if (config.getBatchOutMode().getValue() == batchoutmodeDynamicValue)
        os << "BatchOutMode::DYNAMIC";
    os << "};";
}

LogicalResult CodeEmitter::emitAscMatmulTypeTemplate(Location loc, Type type, bool /*emitAsUnsigned*/)
{
    auto pType = dyn_cast<ascendc::MatmulType>(type);
    if (matmulConfigCounters.empty()) {
        emitError(loc) << "Matmul type emitted outside a C++ scope";
        return failure();
    }
    unsigned configId = matmulConfigCounters.back()++;
    std::string configName = configId == 0 ? "CFG" : "CFG_" + std::to_string(configId);
    std::string staticConfigName = configId == 0 ? "STATIC_CFG" : "STATIC_CFG_" + std::to_string(configId);
    emitMatmulConfig(os, pType.getMatmulConfig(), configName);
    auto config = pType.getMatmulConfig();
    bool useStaticTiling = config.getSingleCoreM().getInt() > 0;
    if (useStaticTiling) {
        os << "constexpr static auto " << staticConfigName << " = " << ascNamespace << "::GetMatmulApiTiling";
        if (failed(emitAscMatmulSimplifiedTemplate(loc, type, false)))
            return failure();
        os << "(" << configName << ");";
    }
    os << "matmul::Matmul";
    os << "<matmul::MatmulType<";
    emitTPosition(os, pType.getSrcAPosition());
    os << ", ";
    emitCubeFormat(os, pType.getCubeFormatA());
    os << ", ";
    if (failed(emitType(loc, pType.getTypeA())))
        return failure();
    os << ", ";
    if (pType.getIsTransA())
        os << "true, ";
    else
        os << "false, ";
    emitLayoutMode(os, pType.getLayoutModeA());
    os << ">, matmul::MatmulType<";
    emitTPosition(os, pType.getSrcBPosition());
    os << ", ";
    emitCubeFormat(os, pType.getCubeFormatB());
    os << ", ";
    if (failed(emitType(loc, pType.getTypeB())))
        return failure();
    os << ", ";
    if (pType.getIsTransB())
        os << "true, ";
    else
        os << "false, ";
    emitLayoutMode(os, pType.getLayoutModeB());
    os << ">, matmul::MatmulType<";
    emitTPosition(os, pType.getSrcCPosition());
    os << ", ";
    emitCubeFormat(os, pType.getCubeFormatC());
    os << ", ";
    if (failed(emitType(loc, pType.getTypeC())))
        return failure();
    os << ", ";
    if (pType.getIsTransC())
        os << "true, ";
    else
        os << "false, ";
    emitLayoutMode(os, pType.getLayoutModeC());
    os << ">, matmul::MatmulType<";
    emitTPosition(os, pType.getSrcBiasPosition());
    os << ", ";
    emitCubeFormat(os, pType.getCubeFormatBias());
    os << ", ";
    if (failed(emitType(loc, pType.getTypeBias())))
        return failure();
    os << ">, ";
    os << (useStaticTiling ? staticConfigName : configName);
    os << ">";
    return success();
}

LogicalResult CodeEmitter::emitAscMatmulSimplifiedTemplate(Location loc, Type type, bool /*emitAsUnsigned*/)
{
    auto pType = dyn_cast<ascendc::MatmulType>(type);
    os << "<matmul::MatmulType<";
    emitTPosition(os, pType.getSrcAPosition());
    os << ", ";
    emitCubeFormat(os, pType.getCubeFormatA());
    os << ", ";
    if (failed(emitType(loc, pType.getTypeA())))
        return failure();
    os << ", ";
    if (pType.getIsTransA())
        os << "true, ";
    else
        os << "false, ";
    emitLayoutMode(os, pType.getLayoutModeA());
    os << ">, matmul::MatmulType<";
    emitTPosition(os, pType.getSrcBPosition());
    os << ", ";
    emitCubeFormat(os, pType.getCubeFormatB());
    os << ", ";
    if (failed(emitType(loc, pType.getTypeB())))
        return failure();
    os << ", ";
    if (pType.getIsTransB())
        os << "true, ";
    else
        os << "false, ";
    emitLayoutMode(os, pType.getLayoutModeB());
    os << ">, matmul::MatmulType<";
    emitTPosition(os, pType.getSrcCPosition());
    os << ", ";
    emitCubeFormat(os, pType.getCubeFormatC());
    os << ", ";
    if (failed(emitType(loc, pType.getTypeC())))
        return failure();
    os << ", ";
    if (pType.getIsTransC())
        os << "true, ";
    else
        os << "false, ";
    emitLayoutMode(os, pType.getLayoutModeC());
    os << ">, matmul::MatmulType<";
    emitTPosition(os, pType.getSrcBiasPosition());
    os << ", ";
    emitCubeFormat(os, pType.getCubeFormatBias());
    os << ", ";
    if (failed(emitType(loc, pType.getTypeBias())))
        return failure();
    os << ">>";
    return success();
}

LogicalResult CodeEmitter::emitAscMatmulType(Location loc, Type type, bool emitAsUnsigned)
{
    return emitAscMatmulTypeTemplate(loc, type, emitAsUnsigned);
}

LogicalResult CodeEmitter::emitIntegerType(IntegerType& iType, Location loc, Type type, bool emitAsUnsigned)
{
    switch (iType.getWidth()) {
        case dtypeBitWidth1:
            return (os << "bool"), success();
        case dtypeBitWidth8:
        case dtypeBitWidth16:
        case dtypeBitWidth32:
        case dtypeBitWidth64:
            if (shouldMapToUnsigned(iType.getSignedness()) || emitAsUnsigned)
                return (os << "uint" << iType.getWidth() << "_t"), success();
            else
                return (os << "int" << iType.getWidth() << "_t"), success();
        default:
            return emitError(loc, "cannot emit integer type ") << type;
    }
}

LogicalResult CodeEmitter::emitFloatType(FloatType& fType, Location loc, Type /*type*/, bool /*emitAsUnsigned*/)
{
    return llvm::TypeSwitch<FloatType, LogicalResult>(fType)
        .Case([this](Float16Type) {
            os << "half";
            return success();
        })
        .Case([this](BFloat16Type) {
            os << "bfloat16_t";
            return success();
        })
        .Case([this](Float32Type) {
            os << "float";
            return success();
        })
        .Case([this](Float64Type) {
            os << "double";
            return success();
        })
        .Default([loc](FloatType type) { return emitError(loc, "cannot emit float type ") << type; });
}

LogicalResult CodeEmitter::emitBaseMemRefType(BaseMemRefType& pType, Location loc, Type /*type*/, bool emitAsUnsigned)
{
    if (auto attr = pType.getMemorySpace()) {
        auto value = static_cast<uint8_t>(cast<IntegerAttr>(attr).getInt());
        if (auto addrSpace = ascendc::symbolizeAddressSpace(value))
            emitAddressSpace(addrSpace.value());
        else
            return failure();
    }
    if (failed(emitType(loc, pType.getElementType(), emitAsUnsigned)))
        return failure();
    os << "*";
    return success();
}

LogicalResult CodeEmitter::emitType(Location loc, Type type, bool emitAsUnsigned)
{
    auto it = emitTypeMapper.find(type.getTypeID());
    if (it != emitTypeMapper.end()) {
        return it->second(loc, type, emitAsUnsigned);
    }
    if (auto iType = dyn_cast<IntegerType>(type)) {
        return emitIntegerType(iType, loc, type, emitAsUnsigned);
    }
    if (auto fType = dyn_cast<FloatType>(type)) {
        return emitFloatType(fType, loc, type, emitAsUnsigned);
    }
    if (auto pType = dyn_cast<BaseMemRefType>(type)) {
        return emitBaseMemRefType(pType, loc, type, emitAsUnsigned);
    }

#define GEN_EMITTER
#include "ascir/API/Types.h.inc"
    return emitError(loc, "cannot emit type ") << type;
}

LogicalResult CodeEmitter::emitAscPyStructType(Location /*loc*/, Type type, bool /*emitAsUnsigned*/)
{
    auto pType = dyn_cast<emitasc::PyStructType>(type);
    os << pType.getNameAttr().getValue();
    return success();
}

LogicalResult CodeEmitter::emitAscDataCopyPadExtParamsType(Location loc, Type type, bool /*emitAsUnsigned*/)
{
    auto ldType = dyn_cast<ascendc::DataCopyPadExtParamsType>(type);
    os << ascNamespace << "::DataCopyPadExtParams<";
    if (failed(emitType(loc, ldType.getElementType())))
        return failure();
    os << '>';
    return success();
}

LogicalResult CodeEmitter::emitAscMrgSortSrcListType(Location loc, Type type, bool /*emitAsUnsigned*/)
{
    auto ldType = dyn_cast<ascendc::MrgSortSrcListType>(type);
    os << ascNamespace << "::MrgSortSrcList<";
    if (failed(emitType(loc, ldType.getElementType())))
        return failure();
    os << '>';
    return success();
}

LogicalResult CodeEmitter::emitAscNdDmaParams(Location loc, Type type, bool /*emitAsUnsigned*/)
{
    auto ldType = dyn_cast<ascendc::NdDmaParamsType>(type);
    os << ascNamespace << "::NdDmaParams<";
    FAIL_OR(emitType(loc, ldType.getType()));
    os << ',' << ldType.getDims() << '>';
    return success();
}

LogicalResult CodeEmitter::emitTypes(Location loc, ArrayRef<Type> types)
{
    switch (types.size()) {
        case 0:
            os << "void";
            return success();
        case dtypeBitWidth1:
            return emitType(loc, types.front());
        default:
            llvm_unreachable("unsupported emission of types array");
    }
}
