/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/VecRemovePad.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Vector layout operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::RemovePadExtendOp op)
{
    auto& os = emitter.ostream();

    // 显式模板实参取**源**的元素类型，与 `printApiExtensionUnary` 同一条。这个包装两端同
    // 类型，所以取哪一端都一样；取源是为了与那一族的写法一致，别在这里开第二种口径。
    auto tensorType = dyn_cast<LocalTensorType>(op.getSrc().getType());
    if (!tensorType) {
        return op->emitOpError(op.getAPIName()) << " requires a local tensor source";
    }

    // 三个尺寸按 C++ 形参表的次序发，每个都 `static_cast<uint64_t>`：包装的形参是
    // `const uint64_t`，而这里的值来自 `index` 的 SSA 值。
    const Value sizes[] = {op.getLastAxisActualSize(), op.getLastAxisSize(), op.getNonLastAxisMerge()};

    // No `AscendC::` qualifier: the extension wrappers are global functions.
    os << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), tensorType.getElementType()));
    os << ">(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc());
    for (Value size : sizes) {
        os << ", static_cast<uint64_t>(" << emitter.getOrCreateName(size) << ")";
    }
    os << ")";

    return success();
}
