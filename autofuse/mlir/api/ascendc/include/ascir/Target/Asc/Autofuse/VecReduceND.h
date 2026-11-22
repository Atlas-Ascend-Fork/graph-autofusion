/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_VEC_REDUCE_ND_H
#define ASCIR_TARGET_ASC_AUTOFUSE_VEC_REDUCE_ND_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// N-dimensional vector reduce operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::ReduceSum2DL2Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::ReduceSumInt32Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::ReduceMax2DL2Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::ReduceMin2DL2Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::ReduceProd2DL2Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::ReduceAny2DL2Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::ReduceAll2DL2Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::ReduceInitOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_VEC_REDUCE_ND_H
