/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You should refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/Transpose.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Vector transpose operations
//===----------------------------------------------------------------------===//

namespace {
constexpr unsigned kBitsPerByte = CHAR_BIT;
constexpr unsigned kMaxElementBytes = sizeof(uint32_t);
constexpr unsigned kMaxTransposeRank = sizeof(uint32_t);

// `RangeType<T>` 的宽度规则（`api_regbase/transpose.h:17-20`）：元素 ≤2 字节用
// `int16_t` 索引，否则 `int32_t`。legacy `GenDimensionParam` 按
// `GetSizeByDataType` 选同一个值，这里按元素位宽取。
std::optional<unsigned> elementTypeWidthInBytes(Type elementType)
{
    if (auto integer = dyn_cast<IntegerType>(elementType)) {
        return integer.getWidth() / kBitsPerByte;
    }
    if (isa<FloatType>(elementType)) {
        return elementType.getIntOrFloatBitWidth() / kBitsPerByte;
    }
    return std::nullopt;
}

} // namespace

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::TransposeExtendRegBaseOp op)
{
    auto& os = emitter.ostream();

    auto tensorType = dyn_cast<LocalTensorType>(op.getDst().getType());
    if (!tensorType) {
        tensorType = dyn_cast<LocalTensorType>(op.getSrc().getType());
    }
    if (!tensorType) {
        return op->emitOpError(op.getAPIName()) << " requires a local tensor operand";
    }
    std::optional<unsigned> elementSize = elementTypeWidthInBytes(tensorType.getElementType());
    if (!elementSize || *elementSize == 0 || *elementSize > kMaxElementBytes) {
        return op->emitOpError(op.getAPIName()) << " requires a 1/2/4-byte element type";
    }
    const llvm::StringRef indexType = *elementSize <= 2 ? "int16_t" : "int32_t";

    const size_t rank = op.getDstDims().size();
    if (rank == 0 || op.getSrcStrides().size() != rank || op.getDstStrides().size() != rank) {
        return op->emitOpError(op.getAPIName()) << " needs one dst dim, src stride and dst stride per segment";
    }

    // 三个尺寸组按花括号列表内联进调用——legacy `GenTransposeDimParam` 的形态，不是
    // 栈数组。每个元素 cast 到 `RangeType<T>` 的索引宽度；值来自 `index` 的 SSA 值。
    auto emitSizeList = [&](OperandRange sizes) {
        os << "{";
        llvm::interleaveComma(sizes, os, [&](Value v) {
            os << "static_cast<" << indexType << ">(" << emitter.getOrCreateName(v) << ")";
        });
        os << "}";
    };

    // 模板实参只发 <inner, total>，T 从 LocalTensor 形参推导——legacy 的 template_params
    // 就是这两个，第三个写出来反而与 legacy 文本不同。
    os << op.getAPIName() << "<" << static_cast<int64_t>(op.getInnerDim()) << ", "
       << static_cast<int64_t>(op.getTotalDim()) << ">(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getTmpBuffer()) << ", ";
    emitSizeList(op.getDstDims());
    os << ", ";
    emitSizeList(op.getSrcStrides());
    os << ", ";
    emitSizeList(op.getDstStrides());
    os << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ConfusionTransposeOp op)
{
    auto& os = emitter.ostream();

    auto tensorType = dyn_cast<LocalTensorType>(op.getSrc().getType());
    if (!tensorType) {
        return op->emitOpError(op.getAPIName()) << " requires a local tensor source";
    }

    // 包装住在 `namespace codegen` 里，前缀手写——与其它扩展包装的全局名不同。
    os << "codegen::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), tensorType.getElementType()));
    os << ">(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << emitter.getOrCreateName(op.getTmpBuffer()) << ", AutoFuseTransposeType::"
       << stringifyAutoFuseTransposeType(op.getTransposeType()) << ", " << emitter.getOrCreateName(op.getApiTilingData())
       << ");\n"
       << "AscendC::PipeBarrier<PIPE_ALL>()";

    return success();
}
