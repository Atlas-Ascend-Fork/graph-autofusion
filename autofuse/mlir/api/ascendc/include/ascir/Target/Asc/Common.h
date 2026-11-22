/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_COMMON_H
#define ASCIR_TARGET_ASC_COMMON_H

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/CodeEmitter.h"
#include "ascir/Target/Asc/TargetUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include <charconv>

#define DEBUG_TYPE "translate-to-ascendc"

#define FAIL_OR(expr) \
    if (failed(expr)) \
    return failure()

#define EXEC_IF_TRUE(condition, expr) \
    if (condition) {                  \
        expr                          \
    }
namespace mlir {

template <bool condition>
using LogicalResultIf = typename std::enable_if_t<condition, LogicalResult>;

template <typename T, typename... AllowedTypes>
using LogicalResultForT = LogicalResultIf<llvm::is_one_of<T, AllowedTypes...>::value>;

// When generating code for an `scf.if` or `cf.cond_br` op no semicolon needs
// to be printed after the closing brace.
// When generating code for an `scf.for op`, printing a trailing semicolon is
// handled within the `printOperation` function.
template <typename OpType>
bool needsSemicolon(const OpType& op)
{
    return !isa<scf::IfOp, scf::ForOp, scf::IndexSwitchOp, scf::YieldOp>(op);
}

template <typename OpType>
LogicalResult isScalarOperation(OpType op)
{
    if (isa<VectorType>(op.getResult().getType())) {
        return op.emitOpError() << "Vector type is not supported.";
    }
    if (isa<TensorType>(op.getResult().getType())) {
        return op.emitOpError() << "Tensor type is not supported.";
    }
    return success();
}

LogicalResult printConstantOp(CodeEmitter& emitter, Operation* operation, Attribute value);

template <typename OpType>
LogicalResult printIsSetMaskCastTemplate(CodeEmitter& emitter, OpType op)
{
    auto& os = emitter.ostream();
    auto dstType = cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    auto srcType = cast<ascendc::LocalTensorType>(op.getSrc1().getType()).getElementType();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstType));
    os << ", ";
    FAIL_OR(emitter.emitType(op.getLoc(), srcType));
    os << ", " << op.getIsSetMask() << ">";
    return success();
}

template <typename OpType>
LogicalResult printIsSetMaskTemplate(CodeEmitter& emitter, OpType op)
{
    auto& os = emitter.ostream();
    auto tensorType = cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), tensorType));
    os << ", " << op.getIsSetMask() << ">";
    return success();
}

template <typename OpType>
auto printMask(CodeEmitter& emitter, OpType op)
{
    static int maskCounter = 0;
    auto uniqueId = std::to_string(maskCounter++);
    auto& os = emitter.ostream();
    auto maskName = (emitter.getOrCreateName(op.getDst()) + "_mask_list" + uniqueId).str();
    os << "uint64_t " << maskName << "[] = {";
    llvm::interleaveComma(op.getMask(), os, [&](Value operand) { os << emitter.getOrCreateName(operand); });
    os << "};\n";

    return maskName;
}

namespace ascendc {

//===----------------------------------------------------------------------===//
// Mask operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::SetVectorMaskL0Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::SetVectorMaskL1Op op);

} // namespace ascendc

} // namespace mlir

mlir::LogicalResult emitOperation(mlir::CodeEmitter& emitter, mlir::Operation& op, bool trailingSemicolon);

#endif // ASCIR_TARGET_ASC_COMMON_H
