/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/VecBinary.h"

using namespace mlir;
using namespace mlir::ascendc;

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::BilinearInterpolationL0Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc0()) << ", " << emitter.getOrCreateName(op.getSrc0Offset()) << ", "
       << emitter.getOrCreateName(op.getSrc1()) << ", " << emitter.getOrCreateName(op.getMask()) << ", "
       << emitter.getOrCreateName(op.getHRepeat()) << ", " << emitter.getOrCreateName(op.getRepeatMode()) << ", "
       << emitter.getOrCreateName(op.getDstBlkStride()) << ", " << emitter.getOrCreateName(op.getVROffset()) << ", "
       << emitter.getOrCreateName(op.getVRepeat()) << ", " << emitter.getOrCreateName(op.getSharedTmpBuffer()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::BilinearInterpolationL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);

    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc0()) << ", " << emitter.getOrCreateName(op.getSrc0Offset()) << ", "
       << emitter.getOrCreateName(op.getSrc1()) << ", " << maskName << ", " << emitter.getOrCreateName(op.getHRepeat())
       << ", " << emitter.getOrCreateName(op.getRepeatMode()) << ", " << emitter.getOrCreateName(op.getDstBlkStride())
       << ", " << emitter.getOrCreateName(op.getVROffset()) << ", " << emitter.getOrCreateName(op.getVRepeat()) << ", "
       << emitter.getOrCreateName(op.getSharedTmpBuffer()) << ")";
    return success();
}
