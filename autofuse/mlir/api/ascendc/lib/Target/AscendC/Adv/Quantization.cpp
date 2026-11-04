/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Adv/Quantization.h"

using namespace mlir;
using namespace mlir::ascendc;

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::QuantOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getSrcTensor().getType().getElementType()));
    os << ", " << op.getIsReuseSource();
    if (auto config = op.getConfig()) {
        os << ", " << emitter.getOrCreateName(config);
    }
    os << ">(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrcTensor());
    if (auto buf = op.getSharedTmpBuffer()) {
        os << ", " << emitter.getOrCreateName(buf);
    }
    os << ", " << emitter.getOrCreateName(op.getScale()) << ", " << emitter.getOrCreateName(op.getOffset());
    if (auto count = op.getCalCount()) {
        os << ", " << emitter.getOrCreateName(count);
    }
    os << ")";
    return success();
}
