/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_CODEEMITTER_H
#define ASCIR_TARGET_ASC_CODEEMITTER_H

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Target/Asc/EmitNameStack.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <unordered_map>
namespace mlir {

static constexpr const char* ascNamespace = "AscendC";
struct CodeEmitter {
    const std::string structFieldNamePrefix = "var";

    static void emitCubeFormat(raw_ostream& os, ascendc::CubeFormat format);

    static void emitTPosition(raw_ostream& os, ascendc::TPosition pos);

    static void emitCO2Layout(raw_ostream& os, ascendc::CO2Layout layout);

    static void emitLayoutMode(raw_ostream& os, ascendc::LayoutMode layout);

    explicit CodeEmitter(raw_ostream& os);

    static void emitMatmulConfig(raw_ostream& os, ascendc::MatmulConfigAttr config, StringRef configName);

    LogicalResult emitAscMatmulSimplifiedTemplate(Location loc, Type type, bool emitAsUnsigned);

    /// Emits attribute or returns failure.
    LogicalResult emitAttribute(Location loc, Attribute attr);

    /// Emits type 'type' or returns failure.
    LogicalResult emitType(Location loc, Type type, bool emitAsUnsigned = false);

    /// Emits array of types as a std::tuple of the emitted types.
    /// - emits void for an empty array;
    /// - emits the type of the only element for arrays of size one;
    /// - emits a std::tuple otherwise;
    LogicalResult emitTypes(Location loc, ArrayRef<Type> types);

    /// Emits a variable declaration for a result of an operation.
    LogicalResult emitVariableDeclaration(OpResult opResult, bool trailingSemicolon);

    /// Emits a label for the block.
    LogicalResult emitLabel(Block& block);

    /// Emits the variable declaration and assignment prefix for 'op'.
    /// - emits separate variable followed by std::tie for multi-valued operation;
    /// - emits single type followed by variable for single result;
    /// - emits nothing if no value produced by op;
    /// Emits final '=' operator where a type is produced. Returns failure if
    /// any result type could not be converted.
    LogicalResult emitAssignPrefix(Operation& op);

    /// Return the existing or a new label of a Block.
    StringRef getOrCreateName(Block& block);

    /// Whether to map an mlir integer to a unsigned integer in C++.
    bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

    /// Emits the operands of the operation. All operands are emitted in order.
    LogicalResult emitOperands(Operation& op);

    /// Return the existing or a new name for a Value.
    StringRef getOrCreateName(Value val);

    /// Return the name of the first value in a range (for Variadic operands).
    StringRef getOrCreateName(Operation::operand_range vals)
    {
        assert(!vals.empty() && "operand range is empty");
        return getOrCreateName(*vals.begin());
    }

    /// Return a string representation of an integer attribute value.
    /// Used by generated code for I32Attr template parameters.
    std::string getOrCreateName(uint32_t val) { return std::to_string(val); }

    /// Emits an address space for MemRefType
    void emitAddressSpace(ascendc::AddressSpace addressSpace);

    /// Emit MatmulType as template
    LogicalResult emitAscMatmulTypeTemplate(Location loc, Type type, bool emitAsUnsigned);

    /// RAII helper function to manage entering/exiting C++ scopes.
    struct Scope {
        Scope(CodeEmitter& emitter)
            : valueMapperScope(emitter.valueMapper), blockMapperScope(emitter.blockMapper), emitter(emitter)
        {
            emitter.nameStack.pushScope();
            emitter.matmulConfigCounters.push_back(0);
        }
        ~Scope()
        {
            emitter.matmulConfigCounters.pop_back();
            emitter.nameStack.popScope();
        }

    private:
        llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
        llvm::ScopedHashTableScope<Block*, std::string> blockMapperScope;
        CodeEmitter& emitter;
    };

    /// Returns wether the Value is assigned to a C++ variable in the scope.
    bool hasValueInScope(Value val);

    // Returns whether a label is assigned to the block.
    bool hasBlockLabel(Block& block);

    /// Returns the output stream.
    raw_indented_ostream& ostream() { return os; };

private:
    using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
    using BlockMapper = llvm::ScopedHashTable<Block*, std::string>;
    using StructGlobalMapper = std::unordered_map<std::string, std::string>;
    using TypeEmitFn = std::function<LogicalResult(Location, Type, bool)>;
    using AttributeEmitFn = std::function<LogicalResult(Location, Attribute)>;
    using EmitTypeMapper = llvm::DenseMap<TypeID, TypeEmitFn>;
    using EmitAttributeMapper = llvm::DenseMap<TypeID, AttributeEmitFn>;
    /// Output stream to emit to.
    raw_indented_ostream os;

    /// Map from value to name of C++ variable that contain the name.
    ValueMapper valueMapper;

    /// Map from block to name of C++ label.
    BlockMapper blockMapper;

    /// Map from mlir struct name to processed name.
    StructGlobalMapper structMapper;

    /// Name stack contains number of values of different type in the current
    /// scope. This is used to declare the names of values in a scope.
    EmitNameStack nameStack;

    EmitTypeMapper emitTypeMapper;

    EmitAttributeMapper emitAttributeMapper;

    /// Number of Matmul configuration declarations emitted in each C++ scope.
    SmallVector<unsigned> matmulConfigCounters;

    void createTypeEmitMapper();

    void createAttributeEmitMapper();

    LogicalResult emitIndexType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitTensorType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitEmitcPointerType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitEmitcOpaqueType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitBaseMemRefType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscTBufType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscTBufPoolType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscQueueType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscQueBindType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscFixpipeParamsType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscGlobalTensorType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscDataCopyPadExtParamsType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscLocalTensorType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscFixpipeParamsC310Type(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscRegTensorType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscMaskRegType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscLocalMemAllocatorType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscPyStructType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscMatmulType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitAscMrgSortSrcListType(Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitIntegerType(IntegerType& iType, Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitFloatType(FloatType& fType, Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitBaseMemRefType(BaseMemRefType& pType, Location loc, Type type, bool emitAsUnsigned);

    LogicalResult emitFloatAttr(Location loc, Attribute attr);

    LogicalResult emitIntegerAttr(Location loc, Attribute attr);

    LogicalResult emitEmitcOpaqueAttr(Location loc, Attribute attr);

    LogicalResult emitSymbolRefAttr(Location loc, Attribute attr);

    LogicalResult emitTypeAttr(Location loc, Attribute attr);

    LogicalResult emitAscNdDmaParams(Location loc, Type type, bool emitAsUnsigned);

    void printInt(const APInt& value, bool isUnsigned);

    void printFloat(const APFloat& value);
};
} // namespace mlir

#endif // ASCIR_TARGET_ASC_CODEEMITTER_H
