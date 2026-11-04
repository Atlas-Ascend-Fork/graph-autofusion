/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Core/ShapeInfo.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// ShapeInfo operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ShapeInfoShapeOp op)
{
    FAIL_OR(emitter.emitAssignPrefix(*op));
    emitter.ostream() << emitter.getOrCreateName(op.getBase()) << ".shape[" << emitter.getOrCreateName(op.getDim())
                      << "]";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ShapeInfoOriginalShapeOp op)
{
    FAIL_OR(emitter.emitAssignPrefix(*op));
    emitter.ostream() << emitter.getOrCreateName(op.getBase()) << ".originalShape["
                      << emitter.getOrCreateName(op.getDim()) << "]";
    return success();
}
