/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/AutofuseVecUnary.h"

using namespace mlir;

LogicalResult mlir::ascendc::printApiExtensionUnary(CodeEmitter& emitter, Operation* op, StringRef apiName, Value dst,
                                                   Value src, Value sharedTmpBuffer, Value calCount, Value constBlock)
{
    auto& os = emitter.ostream();
    // 显式模板实参取的是**源**的元素类型，不是目的地的。多数包装两端同类型，看不出
    // 区别；谓词族（`IsnanExtend<T>(LocalTensor<uint8_t> dst, LocalTensor<T> src, ...)`）
    // 的目的地恒为 uint8，按目的地取会实例化出一个参数对不上的重载。
    auto tensorType = dyn_cast<ascendc::LocalTensorType>(src.getType());
    if (!tensorType) {
        return op->emitOpError(apiName) << " requires a local tensor source";
    }

    // No `AscendC::` qualifier: the extension wrappers are global functions.
    os << apiName << "<";
    FAIL_OR(emitter.emitType(op->getLoc(), tensorType.getElementType()));
    os << ">(" << emitter.getOrCreateName(dst) << ", " << emitter.getOrCreateName(src);
    // 常量 block 排在 tmp buffer **之前**，与两份包装的形参顺序一致
    // （`RsqrtExtend(dst, src, blk, tmp, size)`）。
    if (constBlock) {
        os << ", " << emitter.getOrCreateName(constBlock);
    }
    if (sharedTmpBuffer) {
        os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
    }
    os << ", static_cast<uint32_t>(" << emitter.getOrCreateName(calCount) << "))";
    return success();
}

// 与 `printApiExtensionUnary` 差一个模板实参，而模板实参列表写不出「跳过前面的」，
// 所以单独一条 body——与 `printApiExtensionScalarBinary` 分家的理由完全相同。
//
// 实参表与 legacy 的 `unary_template_attr_api_tmp_call.cpp` 逐字一致：
// `Xxx<dtype, n>(y[off], x[off], tmp, size)`。`n` 直接打十进制字面量，不建
// `arith.constant`——模板位要的是编译期常量，而一个 SSA 值在那里打出来是个变量名。
LogicalResult mlir::ascendc::printApiExtensionUnaryTemplateAttr(CodeEmitter& emitter, Operation* op,
                                                                StringRef apiName, Value dst, Value src,
                                                                Value sharedTmpBuffer, Value calCount,
                                                                int64_t templateArgument)
{
    auto& os = emitter.ostream();
    // 模板实参取**源**的元素类型，与 `printApiExtensionUnary` 同一条理由。
    auto tensorType = dyn_cast<ascendc::LocalTensorType>(src.getType());
    if (!tensorType) {
        return op->emitOpError(apiName) << " requires a local tensor source";
    }

    // No `AscendC::` qualifier: the extension wrappers are reached unqualified,
    // the same way every other `XxxExtend` wrapper is.
    os << apiName << "<";
    FAIL_OR(emitter.emitType(op->getLoc(), tensorType.getElementType()));
    os << ", " << templateArgument << ">(" << emitter.getOrCreateName(dst) << ", " << emitter.getOrCreateName(src)
       << ", " << emitter.getOrCreateName(sharedTmpBuffer) << ", static_cast<uint32_t>("
       << emitter.getOrCreateName(calCount) << "))";
    return success();
}
