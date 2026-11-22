/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/AutofuseVecBinary.h"

using namespace mlir;

LogicalResult mlir::ascendc::printApiExtensionBinary(CodeEmitter& emitter, Operation* op, StringRef apiName, Value dst,
                                                     Value src0, Value src1, Value extraScalar, Value sharedTmpBuffer,
                                                     Value calCount)
{
    // Unlike the compare bodies below, this one needs nothing off the operation
    // itself -- every argument it prints was passed in. The parameter stays for
    // signature symmetry with them.
    (void)op;
    auto& os = emitter.ostream();
    // No `AscendC::` qualifier: the extension wrappers are global functions.
    os << apiName << "(" << emitter.getOrCreateName(dst) << ", " << emitter.getOrCreateName(src0) << ", "
       << emitter.getOrCreateName(src1);
    // `AxpyExtend` puts its `alpha` between the second source and the tmp buffer.
    // It is a plain scalar argument, not a tile, so it needs no offsetting.
    if (extraScalar) {
        os << ", " << emitter.getOrCreateName(extraScalar);
    }
    if (sharedTmpBuffer) {
        os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
    }
    os << ", static_cast<uint32_t>(" << emitter.getOrCreateName(calCount) << "))";
    return success();
}

// 标量形态的二元包装（`Subs` / `Divs` / `TrueDivExtends` / `DivExtends`）。与
// `printApiExtensionBinary` 差两件事，两件都逼着它单独成一条 body：
//
//   - **模板实参必须写出来。** 方向那一位是 bool，排在 dtype 之后，跳不过前面的。
//     `templateResultTensor` 非空时多打一个 dtype——`TrueDivExtends<T, U, LATTER>` 的
//     `U` 是结果类型，而且**即使与 `T` 相同也要写**（少写一个的话 `LATTER` 会落到 `U`
//     的位置上，bool 当类型用，编不过）。`Subs` / `Divs` / `DivExtends` 是两个实参。
//   - **第一个 dtype 取的是 `src` 的元素类型，不是 `dst` 的。** 两者只在 `TrueDivExtends`
//     的 i32→f32 那一行不同，取错在 f16 / f32 上看不出来。
//
// 实参表与 legacy 的 `binary_api_call.cpp` / `truediv_api_call.cpp` 逐字一致：
// `(dst, src, scalar, [tmp,] count)`，标量在 `src` 之后而不是按 AFIR 的操作数顺序——
// 有形状的那个恒是第一个输入实参，方向由模板位上的 bool 说。
LogicalResult mlir::ascendc::printApiExtensionScalarBinary(CodeEmitter& emitter, Operation* op, StringRef apiName,
                                                          Value dst, Value src, Value scalar,
                                                          Value templateResultTensor, bool isScalarLatter,
                                                          Value sharedTmpBuffer, Value calCount)
{
    auto& os = emitter.ostream();
    // No `AscendC::` qualifier: the extension wrappers are global functions.
    os << apiName << "<";
    FAIL_OR(emitter.emitType(op->getLoc(), cast<ascendc::LocalTensorType>(src.getType()).getElementType()));
    if (templateResultTensor) {
        os << ", ";
        FAIL_OR(
            emitter.emitType(op->getLoc(), cast<ascendc::LocalTensorType>(templateResultTensor.getType()).getElementType()));
    }
    os << ", " << (isScalarLatter ? "true" : "false") << ">(" << emitter.getOrCreateName(dst) << ", "
       << emitter.getOrCreateName(src) << ", " << emitter.getOrCreateName(scalar);
    if (sharedTmpBuffer) {
        os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
    }
    os << ", static_cast<uint32_t>(" << emitter.getOrCreateName(calCount) << "))";
    return success();
}

LogicalResult mlir::ascendc::printApiExtensionCompare(CodeEmitter& emitter, Operation* op, StringRef apiName, Value dst,
                                                      Value src0, Value src1, ascendc::CMPMODE cmpMode, Value calCount,
                                                      Value sharedTmpBuffer)
{
    (void)op;
    auto& os = emitter.ostream();
    // No `AscendC::` qualifier, and no explicit template argument: both `InT` and
    // `OutT` are deduced from the tensors, exactly as legacy emits it.
    os << apiName << "(" << emitter.getOrCreateName(dst) << ", " << emitter.getOrCreateName(src0) << ", "
       << emitter.getOrCreateName(src1) << ", CMPMODE::" << ascendc::stringifyEnum(cmpMode)
       << ", static_cast<uint32_t>(" << emitter.getOrCreateName(calCount) << "), "
       << emitter.getOrCreateName(sharedTmpBuffer) << ")";
    return success();
}

LogicalResult mlir::ascendc::printApiExtensionCompareRegBase(CodeEmitter& emitter, Operation* op, StringRef apiName,
                                                             Value dst, Value src0, Value src1,
                                                             ascendc::CMPMODE cmpMode, Value calCount)
{
    auto& os = emitter.ostream();
    os << apiName << "<";
    FAIL_OR(emitter.emitType(op->getLoc(), cast<ascendc::LocalTensorType>(src0.getType()).getElementType()));
    // `dim == 1` is what "contiguous" means here; the two strides follow from it.
    os << ", 1, CMPMODE::" << ascendc::stringifyEnum(cmpMode) << ">(" << emitter.getOrCreateName(dst) << ", "
       << emitter.getOrCreateName(src0) << ", " << emitter.getOrCreateName(src1) << ", {static_cast<uint16_t>("
       << emitter.getOrCreateName(calCount) << ")}, {static_cast<uint16_t>(1)}, {static_cast<uint16_t>(1)})";
    return success();
}

// `BinaryBrcInlineOp`：两处模板实参，一处在包装自己的名字上（`<T>`，由 `dst` 的元素类型
// 推），一处在尾部的函数指针上（`&AscendC::<wrappedApiName><T, false>`）。后者的 `T`
// 与前者同一个值，但必须各打一遍——函数指针那一位不参与整个调用的模板实参推导，写漏了
// 编不过。`false` 是 `isSetMask`，理由见 `OpVecBinary.td` 上这个 op 的注释，不是沿用
// 危险缺省。
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::BinaryBrcInlineOp op)
{
    auto& os = emitter.ostream();
    const Type elementType = cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    os << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op->getLoc(), elementType));
    os << ">(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc0()) << ", "
       << emitter.getOrCreateName(op.getSrc1()) << ", static_cast<int64_t>(" << emitter.getOrCreateName(op.getShape0())
       << "), static_cast<int64_t>(" << emitter.getOrCreateName(op.getShape1()) << "), static_cast<uint8_t>("
       << emitter.getOrCreateName(op.getIsInput0BlockBrc()) << "), static_cast<uint8_t>("
       << emitter.getOrCreateName(op.getIsInput1BlockBrc()) << "), static_cast<int64_t>("
       << emitter.getOrCreateName(op.getFirstAxisVStride()) << "), &AscendC::" << op.getWrappedApiName() << "<";
    FAIL_OR(emitter.emitType(op->getLoc(), elementType));
    os << ", false>)";
    return success();
}
