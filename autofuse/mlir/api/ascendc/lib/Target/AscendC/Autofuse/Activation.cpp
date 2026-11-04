/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/AutofuseActivation.h"

using namespace mlir;
using namespace mlir::ascendc;

// Gelu 与 Sigmoid 逐字同形，只有名字不同；两者的 tmp buffer 与 count 在 ODS 上都是
// optional，所以印法也得一样地按「有就印」来。
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GeluOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc());
    if (auto stb = op.getSharedTmpBuffer()) {
        os << ", " << emitter.getOrCreateName(stb);
    }
    if (auto cc = op.getCalCount()) {
        os << ", static_cast<uint32_t>(" << emitter.getOrCreateName(cc) << ")";
    }
    os << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SigmoidOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc());
    if (auto stb = op.getSharedTmpBuffer()) {
        os << ", " << emitter.getOrCreateName(stb);
    }
    if (auto cc = op.getCalCount()) {
        os << ", static_cast<uint32_t>(" << emitter.getOrCreateName(cc) << ")";
    }
    os << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::SigmoidExtendOp op)
{
    return printApiExtensionUnary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getSharedTmpBuffer(),
                                  op.getCalCount());
}
