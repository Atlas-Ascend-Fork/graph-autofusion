/*
 * Copyright (c) 2025 AISS Group, ISE Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_BASIC_REDUCE_H
#define ASCIR_TARGET_ASC_BASIC_REDUCE_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// BlockReduceMax operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::BlockReduceMaxL1Op op);

//===----------------------------------------------------------------------===//
// BlockReduceMin operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::BlockReduceMinL1Op op);

//===----------------------------------------------------------------------===//
// BlockReduceSum operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::BlockReduceSumL1Op op);

//===----------------------------------------------------------------------===//
// Vector reduce operations
//===----------------------------------------------------------------------===//
LogicalResult printOperation(CodeEmitter& emitter, ascendc::PairReduceSumL1Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::WholeReduceMaxL1Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::WholeReduceMinL1Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::WholeReduceSumL1Op op);

//===----------------------------------------------------------------------===//
// ReduceMax operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::ReduceMaxL1Op op);

//===----------------------------------------------------------------------===//
// ReduceMin operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::ReduceMinL1Op op);

//===----------------------------------------------------------------------===//
// ReduceSum operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::ReduceSumL1Op op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_BASIC_REDUCE_H